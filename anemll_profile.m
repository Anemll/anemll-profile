// anemll_profile.m — ANE CostModel profiler
// Profiles CoreML models via MLComputePlan + Espresso CostModelFeature logging.
// Accepts .mlmodelc, .mlpackage, or base path (auto-detects).
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML -o anemll-profile anemll_profile.m
//
// Usage:
//   anemll-profile model.mlmodelc
//   anemll-profile model.mlpackage
//   anemll-profile /path/to/model    # auto-finds .mlmodelc or .mlpackage
//
#define VERSION "0.4.1"

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <sys/wait.h>
#import <mach/mach_time.h>
#include <fcntl.h>
#include <mach-o/dyld.h>

// ── CostModelFeature parsed entry ──────────────────────────────────────────

typedef struct {
    char name[128];
    char type[64];
    double gFlopCnt;
    double totalMB;
    double mbKernel;
    double mbInput;
    double mbOutput;
    double opsPerByte;
    double workUnitEff;
    double gflops;
    double gbps;
    double runtime;
    int isL2Resident;
    int usedDTree;
    char bound[16];
    int inputCh, outputCh;
    int kernelX, kernelY;
    int inputX, inputY;
    int outputX, outputY;
} CostEntry;

static int parseCostLine(const char *line, CostEntry *e) {
    const char *p = strstr(line, "[CostModelFeature],");
    if (!p) return 0;
    p += 19; // strlen("[CostModelFeature],")

    memset(e, 0, sizeof(*e));
    char buf[2048];
    strncpy(buf, p, sizeof(buf)-1);
    buf[sizeof(buf)-1] = 0;

    char *tokens[128];
    int ntok = 0;
    char *tok = strtok(buf, ",");
    while (tok && ntok < 128) { tokens[ntok++] = tok; tok = strtok(NULL, ","); }
    if (ntok < 4) return 0;

    strncpy(e->name, tokens[0], sizeof(e->name)-1);
    strncpy(e->type, tokens[2], sizeof(e->type)-1);

    for (int i = 3; i < ntok; i++) {
        const char *k = tokens[i];
        // Bound:Memory / Bound:Compute is a key-only token (value embedded in key)
        if (!strncmp(k,"Bound:",6)) { strncpy(e->bound, k+6, sizeof(e->bound)-1); continue; }
        if (i + 1 >= ntok) break;
        const char *v = tokens[i+1];
        if (!strcmp(k,"gFlopCnt")) e->gFlopCnt = atof(v);
        else if (!strcmp(k,"totalMB")) e->totalMB = atof(v);
        else if (!strcmp(k,"mbKernel")) e->mbKernel = atof(v);
        else if (!strcmp(k,"mbInputTensors")) e->mbInput = atof(v);
        else if (!strcmp(k,"mbOutputTensors")) e->mbOutput = atof(v);
        else if (!strcmp(k,"opsPerByte")) e->opsPerByte = atof(v);
        else if (!strcmp(k,"workUnitEfficiency16")) e->workUnitEff = atof(v);
        else if (!strcmp(k,"GFLOP/s")) e->gflops = atof(v);
        else if (!strncmp(k,"GBP/s",5) || !strcmp(k,"GB/s")) e->gbps = atof(v);
        else if (!strcmp(k,"Runtime")) e->runtime = atof(v);
        else if (!strcmp(k,"isL2Resident")) e->isL2Resident = atoi(v);
        else if (!strcmp(k,"UsedDTree")) e->usedDTree = !strcmp(v,"True");
        else if (!strcmp(k,"inputChannelCount")) e->inputCh = atoi(v);
        else if (!strcmp(k,"outputChannelCount")) e->outputCh = atoi(v);
        else if (!strcmp(k,"kernelX")) e->kernelX = atoi(v);
        else if (!strcmp(k,"kernelY")) e->kernelY = atoi(v);
        else if (!strcmp(k,"inputTensorX")) e->inputX = atoi(v);
        else if (!strcmp(k,"inputTensorY")) e->inputY = atoi(v);
        else if (!strcmp(k,"outputTensorX")) e->outputX = atoi(v);
        else if (!strcmp(k,"outputTensorY")) e->outputY = atoi(v);
    }
    return 1;
}

// ── Type aggregation ───────────────────────────────────────────────────────

typedef struct {
    char type[64];
    int count;
    double totalRuntime, totalGFlop, totalMB, weightMB;
    int memBound, compBound;
} TypeAgg;

typedef struct {
    char function[128];
    char name[128];
    char type[64];
    char device[8];
    int index;
    double cost;
    int isConst;
    int aneSupported;
} PlacementEntry;

typedef struct {
    char function[128];
    int startIdx, endIdx;
    int prevAneIdx, nextAneIdx;
    int functionTotalOps;
    int opCount;
    double totalCost;
    double runtimeMs;
    double switchPenaltyMs;
    double totalPenaltyMs;
    int softOps, hardOps;
    int missingRuntimeOps;
    int displayId;
    int hasCPU, hasGPU, hasOther;
    int isInterruption;
    char mainReason[96];
    char firstType[64], lastType[64];
} DeviceIsland;

#define MAX_TYPES  128
#define MAX_ENTRIES 100000
#define MAX_PLACEMENTS 100000
#define MAX_INTERRUPTS 4096

// ── Short type name (strip iosXX. prefix) ──────────────────────────────────

static const char *shortType(const char *type) {
    // "ios18.conv" → "conv", "ios16.reduce_sum" → "reduce_sum"
    if (!strncmp(type, "ios", 3)) {
        const char *dot = strchr(type, '.');
        if (dot) return dot + 1;
    }
    return type;
}

// strip to last N chars with ".." prefix if truncated
static void truncName(char *dst, const char *src, int maxlen) {
    int len = (int)strlen(src);
    if (len <= maxlen) {
        strcpy(dst, src);
    } else {
        dst[0] = '.'; dst[1] = '.';
        strcpy(dst + 2, src + len - (maxlen - 2));
    }
}

static void spanLabel(char *dst, size_t dstSize, int startIdx, int endIdx) {
    if (startIdx == endIdx) snprintf(dst, dstSize, "%d", startIdx);
    else snprintf(dst, dstSize, "%d-%d", startIdx, endIdx);
}

static const char *runDeviceLabel(const DeviceIsland *d, char *dst, size_t dstSize) {
    if (d->hasCPU && d->hasGPU) snprintf(dst, dstSize, "CPU+GPU");
    else if (d->hasCPU) snprintf(dst, dstSize, "CPU");
    else if (d->hasGPU) snprintf(dst, dstSize, "GPU");
    else snprintf(dst, dstSize, "Other");
    return dst;
}

static const char *runPathLabel(const DeviceIsland *d, const char *focusDevice,
                                char *dst, size_t dstSize) {
    char devs[16];
    runDeviceLabel(d, devs, sizeof(devs));
    if (d->prevAneIdx >= 0 && d->nextAneIdx >= 0)
        snprintf(dst, dstSize, "%s->%s->%s", focusDevice, devs, focusDevice);
    else if (d->prevAneIdx >= 0)
        snprintf(dst, dstSize, "%s->%s", focusDevice, devs);
    else if (d->nextAneIdx >= 0)
        snprintf(dst, dstSize, "%s->%s", devs, focusDevice);
    else snprintf(dst, dstSize, "%s", devs);
    return dst;
}

// ── Log capture ────────────────────────────────────────────────────────────

static NSString *g_logPath = nil;
static pid_t g_logPID = 0;

static void startLogCapture(void) {
    g_logPath = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"anemll-profile_%d.log", getpid()]];
    g_logPID = fork();
    if (g_logPID == 0) {
        // Ensure private log data is visible in child too
        setenv("OS_ACTIVITY_DT_MODE", "YES", 1);
        freopen([g_logPath UTF8String], "w", stdout);
        freopen("/dev/null", "w", stderr);
        execlp("/usr/bin/log", "log", "stream",
            "--predicate", "subsystem == \"com.apple.espresso\"",
            "--info", "--debug", "--style", "compact", NULL);
        _exit(1);
    }
    usleep(2000000); // tiny models can finish planning before log stream is attached
}

static void stopLogCapture(void) {
    if (g_logPID > 0) {
        kill(g_logPID, SIGTERM);
        int s; waitpid(g_logPID, &s, 0);
        g_logPID = 0;
    }
    usleep(300000);
}

// ── Resolve model path ────────────────────────────────────────────────────

static NSString *resolveModelPath(const char *arg, NSString **displayName, BOOL *needsCompile) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *input = [NSString stringWithUTF8String:arg];
    while ([input hasSuffix:@"/"]) input = [input substringToIndex:input.length-1];

    *needsCompile = NO;

    // Direct .mlmodelc
    if ([input hasSuffix:@".mlmodelc"] && [fm fileExistsAtPath:input]) {
        *displayName = [input lastPathComponent];
        return input;
    }
    // Direct .mlpackage
    if ([input hasSuffix:@".mlpackage"] && [fm fileExistsAtPath:input]) {
        *displayName = [input lastPathComponent];
        *needsCompile = YES;
        return input;
    }
    // Auto-detect: try .mlmodelc first, then .mlpackage
    NSString *mc = [input stringByAppendingString:@".mlmodelc"];
    NSString *mp = [input stringByAppendingString:@".mlpackage"];
    if ([fm fileExistsAtPath:mc]) {
        *displayName = [mc lastPathComponent];
        return mc;
    }
    if ([fm fileExistsAtPath:mp]) {
        *displayName = [mp lastPathComponent];
        *needsCompile = YES;
        return mp;
    }
    return nil;
}

// ── CLI / async helpers (from branch) ────────────────────────────────────

static NSString * const kANEMLLProfileErrorDomain = @"anemll-profile";

static void printUsage(FILE *stream) {
    fprintf(stream, "Usage:\n");
    fprintf(stream, "  anemll-profile model.mlpackage\n");
    fprintf(stream, "  anemll-profile model.mlmodelc\n");
    fprintf(stream, "  anemll-profile /path/to/model          # auto-detects .mlmodelc or .mlpackage\n");
    fprintf(stream, "  anemll-profile -a model.mlpackage      # include GPU in device assignment\n");
    fprintf(stream, "  anemll-profile --function NAME model   # profile a named function\n");
    fprintf(stream, "  anemll-profile --all-functions model   # profile every function\n");
    fprintf(stream, "  anemll-profile --list-functions model  # list functions and exit\n\n");
    fprintf(stream, "Options:\n");
    fprintf(stream, "  -h, --help                    Show help and exit\n");
    fprintf(stream, "  -v, --version                 Show version and exit\n");
    fprintf(stream, "  -c, --cpu-ane                 CPU + ANE (default)\n");
    fprintf(stream, "  -a, --all                     All devices incl. GPU\n");
    fprintf(stream, "  -f, --function NAME           Profile a specific function\n");
    fprintf(stream, "      --all-functions           Profile every function in the model\n");
    fprintf(stream, "      --list-functions          List available functions and exit\n");
    fprintf(stream, "  -j, --json FILE               Write structured report to JSON file\n");
    fprintf(stream, "      --interrupt-boundary-ms N Heuristic switch cost per ANE boundary (default 300)\n");
}

static BOOL waitForFlag(volatile BOOL *done, NSTimeInterval timeoutSeconds) {
    NSDate *timeout = [NSDate dateWithTimeIntervalSinceNow:timeoutSeconds];
    while (!*done && [[NSDate date] compare:timeout] == NSOrderedAscending)
        [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode
            beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
    return *done;
}

static BOOL supportsFunctionSelection(void) {
    if (@available(macOS 15.0, *)) return YES;
    return NO;
}

static NSError *functionSelectionAvailabilityError(void) {
    return [NSError errorWithDomain:kANEMLLProfileErrorDomain
                               code:1
                           userInfo:@{
                               NSLocalizedDescriptionKey:
                                   @"Multi-function options require macOS 15 or newer."
                           }];
}

static BOOL isMainFunctionName(NSString *functionName) {
    return functionName && [functionName isEqualToString:@"main"];
}

static MLModelAsset *copyModelAssetForCompiledModel(NSString *modelcPath,
                                                    NSError **outError) {
    NSURL *url = [NSURL fileURLWithPath:modelcPath];
    if (@available(macOS 12.0, *)) {
        return [MLModelAsset modelAssetWithURL:url error:outError];
    }
    if (outError) {
        *outError = [NSError errorWithDomain:kANEMLLProfileErrorDomain
                                        code:4
                                    userInfo:@{NSLocalizedDescriptionKey:
                                        @"MLModelAsset requires macOS 12 or newer."}];
    }
    return nil;
}

static BOOL copyFunctionNamesForModelAsset(MLModelAsset *asset,
                                           NSArray<NSString *> **outNames,
                                           NSError **outError) {
    if (!supportsFunctionSelection()) {
        if (outError) *outError = functionSelectionAvailabilityError();
        return NO;
    }

    __block BOOL done = NO;
    __block NSArray<NSString *> *names = nil;
    __block NSError *callbackError = nil;

    if (@available(macOS 15.0, *)) {
        [asset functionNamesWithCompletionHandler:
            ^(NSArray<NSString *> *functionNames, NSError *error) {
                names = [functionNames copy];
                callbackError = error;
                done = YES;
            }];
    }

    if (!waitForFlag(&done, 60)) {
        if (outError) {
            *outError = [NSError errorWithDomain:kANEMLLProfileErrorDomain
                                            code:2
                                        userInfo:@{
                                            NSLocalizedDescriptionKey:
                                                @"Timed out while loading function names."
                                        }];
        }
        return NO;
    }
    if (callbackError) {
        if (outError) *outError = callbackError;
        return NO;
    }

    if (outNames) *outNames = names ?: @[];
    return YES;
}

static MLModel *loadModelForProfiling(NSURL *url,
                                      MLModelAsset *modelAsset,
                                      MLModelConfiguration *configuration,
                                      NSError **outError) {
    if (modelAsset) {
        if (@available(macOS 13.0, *)) {
            __block BOOL done = NO;
            __block MLModel *loadedModel = nil;
            __block NSError *loadError = nil;

            [MLModel loadModelAsset:modelAsset
                      configuration:configuration
                  completionHandler:^(MLModel *model, NSError *error) {
                      loadedModel = model;
                      loadError = error;
                      done = YES;
                  }];

            if (!waitForFlag(&done, 300)) {
                if (outError) {
                    *outError = [NSError errorWithDomain:kANEMLLProfileErrorDomain
                                                    code:3
                                                userInfo:@{
                                                    NSLocalizedDescriptionKey:
                                                        @"Timed out while loading model asset."
                                                }];
                }
                return nil;
            }

            if (outError) *outError = loadError;
            return loadedModel;
        }
    }

    return [MLModel modelWithContentsOfURL:url configuration:configuration error:outError];
}

static BOOL programFunctionHasPlanData(MLComputePlan *plan,
                                       MLModelStructureProgramFunction *fn) {
    NSArray *ops = fn.block.operations;
    for (MLModelStructureProgramOperation *op in ops)
        if ([plan estimatedCostOfMLProgramOperation:op] ||
            [plan computeDeviceUsageForMLProgramOperation:op])
            return YES;
    return NO;
}

static NSArray<NSString *> *resolveProgramFunctionNames(MLComputePlan *plan,
                                                        MLModelStructureProgram *prog,
                                                        NSString *requestedFunctionName,
                                                        NSString **resolvedFunctionName,
                                                        NSString **errorMessage) {
    NSDictionary *funcs = prog.functions;

    if (requestedFunctionName) {
        if (!funcs[requestedFunctionName]) {
            if (errorMessage)
                *errorMessage = [NSString stringWithFormat:
                    @"Function '%@' is not present in the compute plan.",
                    requestedFunctionName];
            return nil;
        }
        if (resolvedFunctionName) *resolvedFunctionName = requestedFunctionName;
        return @[requestedFunctionName];
    }

    if (funcs.count == 1) {
        NSString *onlyName = funcs.allKeys.firstObject;
        if (resolvedFunctionName) *resolvedFunctionName = onlyName;
        return onlyName ? @[onlyName] : @[];
    }

    NSMutableArray<NSString *> *activeNames = [NSMutableArray array];
    for (NSString *fname in funcs) {
        if (programFunctionHasPlanData(plan, funcs[fname]))
            [activeNames addObject:fname];
    }

    if (activeNames.count == 1) {
        if (resolvedFunctionName) *resolvedFunctionName = activeNames[0];
        return [activeNames copy];
    }

    if (errorMessage) {
        *errorMessage = activeNames.count == 0 ?
            @"Unable to determine the active function. Rerun with --function or --all-functions." :
            @"Ambiguous active function. Rerun with --function or --all-functions.";
    }
    return nil;
}

// ── profileModel ──────────────────────────────────────────────────────────
// Full profiling for one model / function. Accepts optional requestedFunctionName
// (nil = single-function model or auto-detect), interruptBoundaryMs for ANE
// interruption heuristic, and jsonPath for optional JSON output.

static int profileModel(NSString *displayName,
                        NSString *modelcPath,
                        MLModelAsset *modelAsset,
                        MLComputeUnits computeUnits,
                        const char *unitsLabel,
                        const char *modelArg,
                        NSString *requestedFunctionName,
                        double interruptBoundaryMs,
                        NSString *jsonPath,
                        BOOL needsCompile,
                        unsigned long long modelSize) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSURL *url = [NSURL fileURLWithPath:modelcPath];

    // ── Clear Espresso cache so CostModelFeature logs are emitted ──
    NSString *processCacheDir = [NSHomeDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"Library/Caches/%@/com.apple.e5rt.e5bundlecache",
            [[NSProcessInfo processInfo] processName]]];
    NSString *globalCacheDir = [NSHomeDirectory() stringByAppendingPathComponent:
        @"Library/Caches/com.apple.e5rt.e5bundlecache"];
    for (NSString *cacheDir in @[processCacheDir, globalCacheDir])
        [fm removeItemAtPath:cacheDir error:nil];

    // ── Start log capture ──────────────────────────────────────────
    startLogCapture();

    // ── Load MLComputePlan ─────────────────────────────────────────
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = computeUnits;
    if (requestedFunctionName) {
        if (@available(macOS 15.0, *)) config.functionName = requestedFunctionName;
    }

    __block BOOL done = NO;
    __block NSString *resolvedFunctionName = requestedFunctionName;
    __block NSString *selectionError = nil;
    __block NSError *planError = nil;
    __block int totalOps = 0, aneOps = 0, cpuOps = 0, gpuOps = 0, constOps = 0;
    __block double totalCost = 0, aneCost = 0, cpuCost = 0, gpuCost = 0;
    __block NSMutableDictionary *opTypeCounts = [NSMutableDictionary dictionary];
    __block NSMutableDictionary *aneOpTypes = [NSMutableDictionary dictionary];
    __block NSMutableDictionary *cpuOpTypes = [NSMutableDictionary dictionary];
    __block NSMutableArray *highCostOps = [NSMutableArray array];
    __block NSMutableArray *cpuOpDetails = [NSMutableArray array];
    __block NSMutableSet *activeLogNames = [NSMutableSet set];
    __block BOOL isNeuralNet = NO;
    __block PlacementEntry *placements = calloc(MAX_PLACEMENTS, sizeof(PlacementEntry));
    __block int nPlacements = 0;

    printf("Loading MLComputePlan ...\n");
    void (^planHandler)(MLComputePlan *, NSError *) =
        ^(MLComputePlan *plan, NSError *error) {
            if (error) {
                planError = error;
                done = YES;
                return;
            }

            MLModelStructure *structure = plan.modelStructure;
            MLModelStructureProgram *prog = structure.program;

            if (prog) {
                NSArray<NSString *> *functionNames =
                    resolveProgramFunctionNames(plan, prog, requestedFunctionName,
                                                &resolvedFunctionName, &selectionError);
                if (!functionNames) {
                    done = YES;
                    return;
                }

                for (NSString *fname in functionNames) {
                    MLModelStructureProgramFunction *fn = prog.functions[fname];
                    NSArray *ops = fn.block.operations;
                    totalOps += (int)ops.count;

                    for (NSUInteger i = 0; i < ops.count; i++) {
                        MLModelStructureProgramOperation *op = ops[i];
                        NSString *opName = op.operatorName;
                        NSString *outName = opName;
                        NSArray *outputs = op.outputs;
                        if ([outputs count] > 0) {
                            NSString *candidate = [outputs[0] name];
                            if ([candidate length] > 0) {
                                outName = candidate;
                                [activeLogNames addObject:candidate];
                            }
                        }
                        MLComputePlanCost *cost = [plan estimatedCostOfMLProgramOperation:op];
                        MLComputePlanDeviceUsage *usage =
                            [plan computeDeviceUsageForMLProgramOperation:op];

                        double w = cost ? cost.weight : 0;
                        if (!cost) constOps++;
                        totalCost += w;

                        opTypeCounts[opName] = @([opTypeCounts[opName] intValue] + 1);

                        NSString *devName = @"?";
                        BOOL aneSupported = NO;
                        NSMutableArray *supList = [NSMutableArray array];
                        NSString *supported = @"";
                        if (usage) {
                            id pref = usage.preferredComputeDevice;
                            NSString *cls = NSStringFromClass([pref class]);
                            if ([cls containsString:@"NeuralEngine"]) {
                                devName = @"ANE"; aneOps++; aneCost += w;
                                aneOpTypes[opName] = @([aneOpTypes[opName] intValue] + 1);
                            } else if ([cls containsString:@"CPU"]) {
                                devName = @"CPU"; cpuOps++; cpuCost += w;
                                cpuOpTypes[opName] = @([cpuOpTypes[opName] intValue] + 1);
                            } else if ([cls containsString:@"GPU"]) {
                                devName = @"GPU"; gpuOps++; gpuCost += w;
                            }
                            for (id dev in usage.supportedComputeDevices) {
                                NSString *dc = NSStringFromClass([dev class]);
                                if ([dc containsString:@"NeuralEngine"]) {
                                    [supList addObject:@"ANE"];
                                    aneSupported = YES;
                                } else if ([dc containsString:@"CPU"]) {
                                    [supList addObject:@"CPU"];
                                } else if ([dc containsString:@"GPU"]) {
                                    [supList addObject:@"GPU"];
                                }
                            }
                            supported = [supList componentsJoinedByString:@","];
                        }

                        if (nPlacements < MAX_PLACEMENTS) {
                            PlacementEntry *p = &placements[nPlacements++];
                            memset(p, 0, sizeof(*p));
                            strncpy(p->function, [fname UTF8String], sizeof(p->function)-1);
                            strncpy(p->name, [outName UTF8String], sizeof(p->name)-1);
                            strncpy(p->type, [opName UTF8String], sizeof(p->type)-1);
                            strncpy(p->device, [devName UTF8String], sizeof(p->device)-1);
                            p->index = (int)i;
                            p->cost = w;
                            p->isConst = cost ? 0 : 1;
                            p->aneSupported = aneSupported ? 1 : 0;
                        }

                        // Collect non-ANE ops with reason
                        if (usage && ![devName isEqualToString:@"ANE"] && cost) {
                            NSString *reason = aneSupported ?
                                @"ANE supported but not preferred" :
                                @"Not supported on ANE";
                            [cpuOpDetails addObject:@{
                                @"name": outName, @"type": opName,
                                @"dev": devName, @"cost": @(w),
                                @"supported": supported,
                                @"reason": reason,
                                @"function": fname,
                                @"idx": @(i)
                            }];
                        }
                        if (w > 0.005) {
                            [highCostOps addObject:@{
                                @"i":@(i), @"op":opName, @"w":@(w), @"dev":devName
                            }];
                        }
                    }
                }
            }

            MLModelStructureNeuralNetwork *nn = structure.neuralNetwork;
            if (nn && !prog) {
                isNeuralNet = YES;
                resolvedFunctionName = resolvedFunctionName ?: @"main";
                NSArray *layers = nn.layers;
                totalOps = (int)layers.count;
                for (NSUInteger i = 0; i < layers.count; i++) {
                    MLModelStructureNeuralNetworkLayer *layer = layers[i];
                    NSString *lt = layer.type;
                    opTypeCounts[lt] = @([opTypeCounts[lt] intValue] + 1);
                    MLComputePlanDeviceUsage *usage =
                        [plan computeDeviceUsageForNeuralNetworkLayer:layer];
                    NSString *devName = @"?";
                    BOOL aneSupported = NO;
                    if (usage) {
                        NSString *cls = NSStringFromClass([[usage preferredComputeDevice] class]);
                        if ([cls containsString:@"NeuralEngine"]) {
                            devName = @"ANE"; aneOps++;
                            aneOpTypes[lt] = @([aneOpTypes[lt] intValue] + 1);
                        } else if ([cls containsString:@"CPU"]) {
                            devName = @"CPU"; cpuOps++;
                            cpuOpTypes[lt] = @([cpuOpTypes[lt] intValue] + 1);
                        } else if ([cls containsString:@"GPU"]) {
                            devName = @"GPU"; gpuOps++;
                        }
                        for (id dev in [usage supportedComputeDevices]) {
                            NSString *dc = NSStringFromClass([dev class]);
                            if ([dc containsString:@"NeuralEngine"]) {
                                aneSupported = YES;
                                break;
                            }
                        }
                    }
                    if (nPlacements < MAX_PLACEMENTS) {
                        PlacementEntry *p = &placements[nPlacements++];
                        memset(p, 0, sizeof(*p));
                        strncpy(p->function, "neuralNetwork", sizeof(p->function)-1);
                        snprintf(p->name, sizeof(p->name), "layer_%lu", (unsigned long)i);
                        strncpy(p->type, [lt UTF8String], sizeof(p->type)-1);
                        strncpy(p->device, [devName UTF8String], sizeof(p->device)-1);
                        p->index = (int)i;
                        p->isConst = 0;
                        p->aneSupported = aneSupported ? 1 : 0;
                    }
                }
            }

            if (!resolvedFunctionName)
                resolvedFunctionName = @"main";
            done = YES;
        };

    BOOL useModelAsset = NO;
    if (modelAsset) {
        if (@available(macOS 14.4, *)) useModelAsset = YES;
    }
    if (useModelAsset) {
        if (@available(macOS 14.4, *)) {
            [MLComputePlan loadModelAsset:modelAsset
                            configuration:config
                        completionHandler:planHandler];
        }
    } else {
        [MLComputePlan loadContentsOfURL:url
                            configuration:config
                        completionHandler:planHandler];
    }

    if (!waitForFlag(&done, 300)) {
        stopLogCapture();
        if (g_logPath) { [fm removeItemAtPath:g_logPath error:nil]; g_logPath = nil; }
        free(placements);
        fprintf(stderr, "Timeout\n");
        return 1;
    }

    sleep(1);
    stopLogCapture();

    if (planError || selectionError) {
        if (g_logPath) { [fm removeItemAtPath:g_logPath error:nil]; g_logPath = nil; }
        free(placements);
        if (planError)
            fprintf(stderr, "MLComputePlan error: %s\n",
                [[planError localizedDescription] UTF8String]);
        else
            fprintf(stderr, "Error: %s\n", [selectionError UTF8String]);
        return 1;
    }

    // ── Parse CostModelFeature logs ────────────────────────────────
    NSString *logContent = [NSString stringWithContentsOfFile:g_logPath
        encoding:NSUTF8StringEncoding error:nil];
    NSArray *logLines = [logContent componentsSeparatedByString:@"\n"];

    CostEntry *entries = calloc(MAX_ENTRIES, sizeof(CostEntry));
    int nEntries = 0;
    NSMutableSet *seenNames = [NSMutableSet set];
    CostEntry *unique = calloc(MAX_ENTRIES, sizeof(CostEntry));
    int nUnique = 0;

    // Parse "Unsupported op" reasons
    NSMutableDictionary *unsupportedReasons = [NSMutableDictionary dictionary];

    for (NSString *line in logLines) {
        if (nEntries < MAX_ENTRIES) {
            CostEntry e;
            if (parseCostLine([line UTF8String], &e)) {
                NSString *ns = [NSString stringWithUTF8String:e.name];
                BOOL keepEntry = (activeLogNames.count == 0 ||
                                  [activeLogNames containsObject:ns]);
                if (keepEntry) {
                    entries[nEntries++] = e;
                    if (![seenNames containsObject:ns]) {
                        [seenNames addObject:ns];
                        unique[nUnique++] = e;
                    }
                }
            }
        }
        // Parse unsupported op reasons
        NSRange r = [line rangeOfString:@"Unsupported op "];
        if (r.location != NSNotFound) {
            NSString *rest = [line substringFromIndex:r.location + r.length];
            NSScanner *sc = [NSScanner scannerWithString:rest];
            int idx = 0;
            if ([sc scanInt:&idx]) {
                [sc scanString:@"(" intoString:nil];
                NSString *type = nil;
                [sc scanUpToString:@")" intoString:&type];
                [sc scanString:@"): " intoString:nil];
                if ([sc scanLocation] < rest.length) {
                    NSString *reason = [rest substringFromIndex:[sc scanLocation]];
                    if (type && reason.length > 0) {
                        NSMutableSet *reasons = unsupportedReasons[type];
                        if (!reasons) {
                            reasons = [NSMutableSet set];
                            unsupportedReasons[type] = reasons;
                        }
                        [reasons addObject:reason];
                    }
                }
            }
        }
    }

    NSMutableDictionary *runtimeByName = [NSMutableDictionary dictionary];
    for (int i = 0; i < nUnique; i++) {
        NSString *name = [NSString stringWithUTF8String:unique[i].name];
        runtimeByName[name] = @(unique[i].runtime);
    }

    const char *focusDevice =
        (computeUnits == MLComputeUnitsAll && gpuOps > 0 &&
         (aneOps == 0 || gpuCost > aneCost || (gpuCost == aneCost && gpuOps >= aneOps)))
        ? "GPU" : "ANE";

    NSString* (^reasonForPlacement)(const PlacementEntry *) = ^NSString *(const PlacementEntry *p) {
        if (p->aneSupported) return @"ANE supported but not preferred";
        NSMutableSet *reasons = unsupportedReasons[[NSString stringWithUTF8String:p->type]];
        if (reasons && [reasons count] > 0) {
            NSArray *sorted = [[reasons allObjects] sortedArrayUsingSelector:@selector(compare:)];
            return sorted[0];
        }
        return @"Not supported on ANE";
    };

    // ── Analyze accelerator graph interruptions ──────────────────
    DeviceIsland *islands = calloc(MAX_INTERRUPTS, sizeof(DeviceIsland));
    int nIslands = 0;
    int interruptionCount = 0, edgeRunCount = 0;
    int interruptedOps = 0, interruptedSoftOps = 0, interruptedHardOps = 0;
    int interruptedMissingRuntimeOps = 0;
    int totalNonAneOps = 0;
    double interruptedCost = 0, totalNonAneCost = 0;
    double interruptedRuntimeMs = 0, interruptedSwitchTaxMs = 0, interruptedPenaltyMs = 0;

    for (int base = 0; base < nPlacements; ) {
        int end = base + 1;
        while (end < nPlacements &&
               !strcmp(placements[end].function, placements[base].function)) end++;

        int functionTotalOps = placements[end - 1].index + 1;
        int lastAneIdx = -1;
        BOOL inRun = NO;
        DeviceIsland run;
        memset(&run, 0, sizeof(run));

        for (int i = base; i < end; i++) {
            PlacementEntry *p = &placements[i];
            if (p->isConst || !strcmp(p->device, "?")) continue;

            if (!strcmp(p->device, focusDevice)) {
                if (inRun) {
                    run.nextAneIdx = p->index;
                    run.isInterruption = run.prevAneIdx >= 0;
                    run.switchPenaltyMs =
                        (run.prevAneIdx >= 0 ? interruptBoundaryMs : 0.0) +
                        (run.nextAneIdx >= 0 ? interruptBoundaryMs : 0.0);
                    run.totalPenaltyMs = run.switchPenaltyMs + run.runtimeMs;
                    if (nIslands < MAX_INTERRUPTS) islands[nIslands++] = run;
                    totalNonAneOps += run.opCount;
                    totalNonAneCost += run.totalCost;
                    if (run.isInterruption) {
                        interruptionCount++;
                        interruptedOps += run.opCount;
                        interruptedCost += run.totalCost;
                        interruptedSoftOps += run.softOps;
                        interruptedHardOps += run.hardOps;
                        interruptedMissingRuntimeOps += run.missingRuntimeOps;
                        interruptedRuntimeMs += run.runtimeMs;
                        interruptedSwitchTaxMs += run.switchPenaltyMs;
                        interruptedPenaltyMs += run.totalPenaltyMs;
                    } else {
                        edgeRunCount++;
                    }
                    inRun = NO;
                    memset(&run, 0, sizeof(run));
                }
                lastAneIdx = p->index;
                continue;
            }

            if (!inRun) {
                memset(&run, 0, sizeof(run));
                strncpy(run.function, p->function, sizeof(run.function)-1);
                run.startIdx = p->index;
                run.endIdx = p->index;
                run.prevAneIdx = lastAneIdx;
                run.nextAneIdx = -1;
                run.functionTotalOps = functionTotalOps;
                strncpy(run.firstType, p->type, sizeof(run.firstType)-1);
                strncpy(run.lastType, p->type, sizeof(run.lastType)-1);
                inRun = YES;
            }

            run.endIdx = p->index;
            run.opCount++;
            run.totalCost += p->cost;
            if (p->aneSupported) run.softOps++;
            else run.hardOps++;
            if (!strcmp(p->device, "CPU")) run.hasCPU = 1;
            else if (!strcmp(p->device, "GPU")) run.hasGPU = 1;
            else run.hasOther = 1;
            strncpy(run.lastType, p->type, sizeof(run.lastType)-1);

            NSNumber *rt = runtimeByName[[NSString stringWithUTF8String:p->name]];
            if (rt) run.runtimeMs += [rt doubleValue];
            else run.missingRuntimeOps++;

            NSString *reason = reasonForPlacement(p);
            const char *rstr = [reason UTF8String];
            if (!p->aneSupported) {
                if (!run.mainReason[0] ||
                    !strcmp(run.mainReason, "ANE supported but not preferred")) {
                    strncpy(run.mainReason, rstr, sizeof(run.mainReason)-1);
                } else if (strcmp(run.mainReason, rstr) != 0) {
                    char mixed[sizeof(run.mainReason)];
                    if (!strstr(run.mainReason, "(+mixed)")) {
                        snprintf(mixed, sizeof(mixed), "%s (+mixed)", run.mainReason);
                        strncpy(run.mainReason, mixed, sizeof(run.mainReason)-1);
                    }
                }
            } else if (!run.mainReason[0]) {
                strncpy(run.mainReason, rstr, sizeof(run.mainReason)-1);
            }
        }

        if (inRun) {
            run.isInterruption = 0;
            run.switchPenaltyMs =
                (run.prevAneIdx >= 0 ? interruptBoundaryMs : 0.0) +
                (run.nextAneIdx >= 0 ? interruptBoundaryMs : 0.0);
            run.totalPenaltyMs = run.switchPenaltyMs + run.runtimeMs;
            if (nIslands < MAX_INTERRUPTS) islands[nIslands++] = run;
            totalNonAneOps += run.opCount;
            totalNonAneCost += run.totalCost;
            edgeRunCount++;
        }

        base = end;
    }

    for (int i = 0; i < nIslands-1; i++) {
        for (int j = i+1; j < nIslands; j++) {
            BOOL swap = NO;
            if (islands[j].isInterruption != islands[i].isInterruption) {
                swap = islands[j].isInterruption > islands[i].isInterruption;
            } else if (islands[j].totalPenaltyMs != islands[i].totalPenaltyMs) {
                swap = islands[j].totalPenaltyMs > islands[i].totalPenaltyMs;
            } else if (islands[j].runtimeMs != islands[i].runtimeMs) {
                swap = islands[j].runtimeMs > islands[i].runtimeMs;
            } else if (islands[j].startIdx < islands[i].startIdx) {
                swap = YES;
            }
            if (swap) {
                DeviceIsland t = islands[i];
                islands[i] = islands[j];
                islands[j] = t;
            }
        }
    }

    for (int i = 0, rank = 1; i < nIslands; i++) {
        islands[i].displayId = islands[i].isInterruption ? rank++ : 0;
    }

    // ── Aggregate by type ──────────────────────────────────────────
    TypeAgg types[MAX_TYPES];
    int nTypes = 0;
    for (int i = 0; i < nUnique; i++) {
        CostEntry *e = &unique[i];
        int f = -1;
        for (int j = 0; j < nTypes; j++)
            if (!strcmp(types[j].type, e->type)) { f = j; break; }
        if (f < 0) {
            f = nTypes++;
            memset(&types[f], 0, sizeof(TypeAgg));
            strncpy(types[f].type, e->type, sizeof(types[f].type)-1);
        }
        types[f].count++;
        types[f].totalRuntime += e->runtime;
        types[f].totalGFlop += e->gFlopCnt;
        types[f].totalMB += e->totalMB;
        if (strstr(e->type, "conv") || strstr(e->type, "matmul"))
            types[f].weightMB += e->mbKernel;
        else if (strstr(e->type, "constexpr_lut"))
            types[f].weightMB += e->mbInput;
        if (!strcmp(e->bound,"Memory")) types[f].memBound++;
        else if (!strcmp(e->bound,"Compute")) types[f].compBound++;
    }
    // Sort descending by runtime
    for (int i = 0; i < nTypes-1; i++)
        for (int j = i+1; j < nTypes; j++)
            if (types[j].totalRuntime > types[i].totalRuntime) {
                TypeAgg t = types[i]; types[i] = types[j]; types[j] = t;
            }

    double grandRT = 0, grandGF = 0;
    for (int i = 0; i < nTypes; i++) {
        grandRT += types[i].totalRuntime;
        grandGF += types[i].totalGFlop;
    }

    BOOL noCostEntries = NO;
    BOOL logDataMasked = NO;
    NSString *measurementStatus = @"not_run";
    NSString *measurementError = nil;
    double measuredSteadyMs = 0, measuredAvgMs = 0, measuredIterPerSec = 0;
    double measuredGFLOPs = 0, measuredTOPS = 0, measuredWeightBW = 0, measuredSpeedup = 0;
    int measuredRuns = 0;

    // Sort unique entries by runtime for top-N
    for (int i = 0; i < nUnique-1; i++)
        for (int j = i+1; j < nUnique; j++)
            if (unique[j].runtime > unique[i].runtime) {
                CostEntry t = unique[i]; unique[i] = unique[j]; unique[j] = t;
            }

    // ═══════════════════════════════════════════════════════════════
    //  PRINT REPORT
    // ═══════════════════════════════════════════════════════════════

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  ANE CostModel Report: %s\n", [displayName UTF8String]);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    printf("  Model size:   %.1f MB\n", modelSize / 1048576.0);
    printf("  Format:       %s\n", isNeuralNet ? "Neural Network" : "ML Program");
    if (resolvedFunctionName)
        printf("  Function:     %s\n", [resolvedFunctionName UTF8String]);
    printf("  Compute:      %s\n", unitsLabel);
    printf("  Total ops:    %d\n", totalOps);
    if (totalCost > 0) {
        printf("  ANE ops:      %d (%.1f%% of cost)\n", aneOps, aneCost/totalCost*100);
        printf("  CPU ops:      %d (%.1f%% of cost)\n", cpuOps, cpuCost/totalCost*100);
        if (gpuOps) printf("  GPU ops:      %d (%.1f%% of cost)\n", gpuOps, gpuCost/totalCost*100);
    } else {
        printf("  ANE ops:      %d\n", aneOps);
        printf("  CPU ops:      %d\n", cpuOps);
        if (gpuOps) printf("  GPU ops:      %d\n", gpuOps);
    }
    if (constOps) printf("  Const ops:    %d (no cost)\n", constOps);
    printf("  CostModel:    %d entries, %d unique ops\n", nEntries, nUnique);

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  %s Graph Interruptions\n", focusDevice);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    printf("  Switch model:           %.1f ms per boundary (%.1f ms full interruption)\n",
        interruptBoundaryMs, interruptBoundaryMs * 2.0);
    printf("  Non-%s runs:            %d\n", focusDevice, nIslands);
    printf("  %s graph interruptions: %d\n", focusDevice, interruptionCount);
    if (edgeRunCount > 0)
        printf("  Edge runs ignored:      %d\n", edgeRunCount);
    if (totalNonAneOps > 0) {
        double pctOps = (double)interruptedOps / totalNonAneOps * 100.0;
        printf("  Interrupted ops:        %d / %d non-%s ops (%.1f%%)\n",
            interruptedOps, totalNonAneOps, focusDevice, pctOps);
    }
    if (totalNonAneCost > 0) {
        double pctNonAne = interruptedCost / totalNonAneCost * 100.0;
        double pctTotal = totalCost > 0 ? interruptedCost / totalCost * 100.0 : 0.0;
        printf("  Interrupted cost:       %.4f (%.1f%% of non-%s, %.1f%% of total)\n",
            interruptedCost, pctNonAne, focusDevice, pctTotal);
    }
    if (interruptionCount > 0) {
        printf("  Estimated switch tax:   %.1f ms/prediction\n", interruptedSwitchTaxMs);
        printf("  Estimated island time:  %.3f ms/prediction\n", interruptedRuntimeMs);
        printf("  Estimated total tax:    %.3f ms/prediction\n", interruptedPenaltyMs);
        printf("  Best-case savings:      up to %.3f ms/prediction\n", interruptedPenaltyMs);
    }
    if (interruptedMissingRuntimeOps > 0)
        printf("  Runtime coverage:       %d interrupted ops missing CostModelFeature timing\n",
            interruptedMissingRuntimeOps);
    if (interruptionCount > 0)
        printf("  Soft vs hard ops:       %d soft, %d hard\n",
            interruptedSoftOps, interruptedHardOps);

    if (interruptionCount > 0) {
        printf("\n  Top interruption islands by estimated latency tax\n\n");
        printf("  %-3s %6s %-16s %-9s %-17s %5s %8s %8s %8s %5s %5s %s\n",
            "#", "At", "Function", "Span", "Path", "Ops",
            "Switch", "Island", "Total", "Soft", "Hard", "Reason");
        printf("  %-3s %6s %-16s %-9s %-17s %5s %8s %8s %8s %5s %5s %s\n",
            "───", "──────", "────────────────", "─────────", "─────────────────",
            "─────", "────────", "────────", "────────", "─────", "─────",
            "────────────────────────────────────────");

        int shown = 0;
        for (int i = 0; i < nIslands && shown < 10; i++) {
            DeviceIsland *d = &islands[i];
            if (!d->isInterruption) continue;

            char func[17], span[10], path[24], reason[41], at[8];
            truncName(func, d->function, 16);
            spanLabel(span, sizeof(span), d->startIdx, d->endIdx);
            truncName(reason, d->mainReason[0] ? d->mainReason : "Unknown reason", 40);
            snprintf(at, sizeof(at), "%5.1f%%",
                d->functionTotalOps > 0 ?
                (((d->startIdx + d->endIdx) / 2.0) + 0.5) / d->functionTotalOps * 100.0 : 0.0);
            runPathLabel(d, focusDevice, path, sizeof(path));
            printf("  %-3d %6s %-16s %-9s %-17s %5d %8.1f %8.3f %8.3f %5d %5d %s\n",
                d->displayId, at, func, span, path, d->opCount,
                d->switchPenaltyMs, d->runtimeMs, d->totalPenaltyMs,
                d->softOps, d->hardOps, reason);
            shown++;
        }
    } else {
        printf("  No %s graph interruptions detected.\n", focusDevice);
    }

    printf("\n  Timeline by function (const/? ops omitted):\n");
    for (int base = 0; base < nPlacements; ) {
        int end = base + 1;
        while (end < nPlacements &&
               !strcmp(placements[end].function, placements[base].function)) end++;

        char func[17];
        truncName(func, placements[base].function, 16);
        printf("    %-16s ", func);
        int lineLen = 21;
        BOOL firstSegment = YES;

        for (int i = base; i < end; ) {
            while (i < end &&
                   (placements[i].isConst || !strcmp(placements[i].device, "?"))) i++;
            if (i >= end) break;

            PlacementEntry *p = &placements[i];
            int runStart = p->index;
            int runEnd = p->index;
            int j = i + 1;
            while (j < end) {
                PlacementEntry *q = &placements[j];
                if (q->isConst || !strcmp(q->device, "?")) { j++; continue; }
                if (strcmp(q->device, p->device)) break;
                runEnd = q->index;
                j++;
            }

            char span[16], segment[96];
            spanLabel(span, sizeof(span), runStart, runEnd);
            if (!strcmp(p->device, focusDevice)) {
                snprintf(segment, sizeof(segment), "%s %s", focusDevice, span);
            } else {
                DeviceIsland *match = NULL;
                for (int k = 0; k < nIslands; k++) {
                    if (!strcmp(islands[k].function, p->function) &&
                        islands[k].startIdx == runStart &&
                        islands[k].endIdx == runEnd) {
                        match = &islands[k];
                        break;
                    }
                }

                char devs[16];
                if (match) runDeviceLabel(match, devs, sizeof(devs));
                else strncpy(devs, p->device, sizeof(devs)-1), devs[sizeof(devs)-1] = 0;

                if (match && match->isInterruption) {
                    snprintf(segment, sizeof(segment), "INT#%d %s %s",
                        match->displayId, devs, span);
                } else if (match && match->prevAneIdx < 0 && match->nextAneIdx < 0) {
                    snprintf(segment, sizeof(segment), "all-%s %s", devs, span);
                } else if (match && match->prevAneIdx < 0) {
                    snprintf(segment, sizeof(segment), "lead-%s %s", devs, span);
                } else if (match) {
                    snprintf(segment, sizeof(segment), "tail-%s %s", devs, span);
                } else {
                    snprintf(segment, sizeof(segment), "%s %s", devs, span);
                }
            }

            int segLen = (int)strlen(segment);
            if (!firstSegment) {
                if (lineLen + 3 + segLen > 110) {
                    printf("\n                      ");
                    lineLen = 22;
                } else {
                    printf(" | ");
                    lineLen += 3;
                }
            }
            printf("%s", segment);
            lineLen += segLen;
            firstSegment = NO;
            i = j;
        }
        printf("\n");
        base = end;
    }

    if (edgeRunCount > 0) {
        printf("\n  Ignored edge non-ANE runs:\n");
        int shown = 0;
        for (int i = 0; i < nIslands && shown < 5; i++) {
            DeviceIsland *d = &islands[i];
            if (d->isInterruption) continue;

            char func[17], first[13], last[13], devs[16], edge[6], span[16];
            truncName(func, d->function, 16);
            truncName(first, shortType(d->firstType), 12);
            truncName(last, shortType(d->lastType), 12);
            runDeviceLabel(d, devs, sizeof(devs));
            if (d->prevAneIdx < 0 && d->nextAneIdx < 0) strcpy(edge, "all");
            else if (d->prevAneIdx < 0) strcpy(edge, "lead");
            else strcpy(edge, "trail");
            spanLabel(span, sizeof(span), d->startIdx, d->endIdx);
            printf("    %-4s %-16s %-11s %-8s %5d %8.1f ms  %s -> %s\n",
                edge, func, span, devs, d->opCount, d->switchPenaltyMs, first, last);
            shown++;
        }
    }

    if (nUnique == 0) {
        printf("\n  ⚠ No CostModelFeature entries captured.\n");
        printf("  Try clearing cache: rm -rf ~/Library/Caches/anemll-profile/com.apple.e5rt*\n");
        noCostEntries = YES;
        goto write_json;
    }

    // Detect <private> masking
    if (nUnique == 1 && nEntries > 10 && strstr(unique[0].name, "private")) {
        printf("\n  ⚠ Log data is masked (<private>). Run with:\n");
        printf("    OS_ACTIVITY_DT_MODE=YES anemll-profile %s\n", modelArg);
        logDataMasked = YES;
        goto write_json;
    }

    // ── Op-Type Runtime Breakdown ──────────────────────────────────
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  Op-Type Runtime Breakdown\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    printf("  %-32s %5s %10s %10s %8s %8s %6s %s\n",
        "Op Type", "Count", "ms/op", "Total ms", "GFLOP", "GB/s", "Share", "Bound");
    printf("  %-32s %5s %10s %10s %8s %8s %6s %s\n",
        "────────────────────────────────", "─────", "──────────",
        "──────────", "────────", "────────", "──────", "──────");

    double grandMB = 0, grandWeightMB = 0;
    for (int i = 0; i < nTypes; i++) {
        grandMB += types[i].totalMB;
        grandWeightMB += types[i].weightMB;
    }

    for (int i = 0; i < nTypes; i++) {
        double pct = grandRT > 0 ? types[i].totalRuntime / grandRT * 100 : 0;
        double gbps = types[i].totalRuntime > 0 ?
            types[i].totalMB / types[i].totalRuntime : 0;
        const char *b = types[i].compBound > 0 ? "Comp" :
                       (types[i].memBound > 0 ? "Mem" : "?");
        printf("  %-32s %5d %10.6f %10.3f %8.4f %8.2f %5.1f%% %s\n",
            shortType(types[i].type), types[i].count,
            types[i].totalRuntime / types[i].count,
            types[i].totalRuntime, types[i].totalGFlop, gbps, pct, b);
    }
    double grandGBs = grandRT > 0 ? grandMB / grandRT : 0;
    printf("\n  %-32s       %10s %10.3f %8.4f %8.2f\n",
        "TOTAL (sum, sequential)", "", grandRT, grandGF, grandGBs);
    if (grandWeightMB > 0)
        printf("  Weights:   %.1f MB (conv/matmul kernels + LUT compressed)\n",
            grandWeightMB);

    // ── Measured prediction time ─────────────────────────────────
    {
        printf("\n  Measuring actual prediction time...\n");
        // Suppress stderr noise during model load
        int saved_stderr = dup(STDERR_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        dup2(devnull, STDERR_FILENO);
        close(devnull);

        NSError *loadErr = nil;
        MLModelConfiguration *mcfg = [[MLModelConfiguration alloc] init];
        mcfg.computeUnits = computeUnits;
        if (requestedFunctionName) {
            if (@available(macOS 15.0, *)) mcfg.functionName = requestedFunctionName;
        }
        MLModel *model = loadModelForProfiling(url, modelAsset, mcfg, &loadErr);

        // Restore stderr
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);
        if (loadErr) {
            measurementStatus = @"load_error";
            measurementError = [loadErr localizedDescription];
            printf("  ⚠ Could not load model: %s\n",
                [[loadErr localizedDescription] UTF8String]);
        } else {
            // Build dummy input from model description
            MLModelDescription *desc = [model modelDescription];
            NSDictionary *inputDescs = [desc inputDescriptionsByName];
            NSMutableDictionary *inputDict = [NSMutableDictionary dictionary];
            BOOL inputOK = YES;
            BOOL hasState = NO;

            for (NSString *name in inputDescs) {
                MLFeatureDescription *fd = inputDescs[name];
                if (fd.type == MLFeatureTypeMultiArray) {
                    MLMultiArrayConstraint *c = fd.multiArrayConstraint;
                    NSError *arrErr = nil;
                    MLMultiArray *arr = [[MLMultiArray alloc]
                        initWithShape:c.shape dataType:c.dataType error:&arrErr];
                    if (arrErr) { inputOK = NO; break; }
                    inputDict[name] = [MLFeatureValue featureValueWithMultiArray:arr];
                } else if (fd.type == MLFeatureTypeState) {
                    hasState = YES;
                } else {
                    inputOK = NO; break;
                }
            }

            // Check stateDescriptionsByName for stateful models
            if (@available(macOS 15.0, *)) {
                NSDictionary *stateDescs = [desc stateDescriptionsByName];
                if (stateDescs.count > 0) hasState = YES;
            }

            if (inputOK) {
                MLDictionaryFeatureProvider *provider =
                    [[MLDictionaryFeatureProvider alloc]
                        initWithDictionary:inputDict error:nil];
                MLState *state = hasState ? [model newState] : nil;

                mach_timebase_info_data_t tbi;
                mach_timebase_info(&tbi);

                // Prediction helper: handles both stateful and stateless models
                void (^predict)(void) = ^{
                    if (state)
                        [model predictionFromFeatures:provider
                            usingState:state error:nil];
                    else
                        [model predictionFromFeatures:provider error:nil];
                };

                // Warmup: first run triggers compilation
                NSError *predErr = nil;
                if (state)
                    [model predictionFromFeatures:provider
                        usingState:state error:&predErr];
                else
                    [model predictionFromFeatures:provider error:&predErr];

                if (predErr) {
                    measurementStatus = @"prediction_failed";
                    measurementError = [predErr localizedDescription];
                    printf("  ⚠ Prediction failed: %s\n",
                        [[predErr localizedDescription] UTF8String]);
                } else {
                    // Second run = steady-state (compilation done)
                    uint64_t tw0 = mach_absolute_time();
                    predict();
                    uint64_t tw1 = mach_absolute_time();
                    double steadyMs = (double)(tw1 - tw0) * tbi.numer / tbi.denom / 1e6;
                    measuredSteadyMs = steadyMs;

                    // Detect ANE compile failure: steady-state >> estimate
                    if (grandRT > 0 && steadyMs > grandRT * 3) {
                        measurementStatus = @"ane_compile_fallback_detected";
                        printf("  ⚠ ANE compilation likely failed — model fell back to CPU\n");
                        printf("  CPU fallback: %.1f ms (vs %.1f ms estimated on ANE)\n",
                            steadyMs, grandRT);
                    } else {
                        measurementStatus = @"ok";
                        // Adaptive run count
                        int nRuns = steadyMs > 1000 ? 1 : (steadyMs > 100 ? 3 : 10);
                        measuredRuns = nRuns;

                        // Extra warmup for fast models
                        if (nRuns >= 10) predict();

                        // Timed runs
                        uint64_t t0 = mach_absolute_time();
                        for (int i = 0; i < nRuns; i++) predict();
                        uint64_t t1 = mach_absolute_time();

                        double totalNs = (double)(t1 - t0) * tbi.numer / tbi.denom;
                        double avgMs = totalNs / nRuns / 1e6;
                        measuredAvgMs = avgMs;
                        measuredIterPerSec = avgMs > 0 ? 1000.0 / avgMs : 0;
                        measuredGFLOPs = avgMs > 0 ? grandGF / avgMs * 1000 : 0;
                        measuredTOPS = avgMs > 0 ? grandGF / avgMs : 0;
                        measuredWeightBW = avgMs > 0 ? grandWeightMB / avgMs : 0;
                        measuredSpeedup = avgMs > 0 ? grandRT / avgMs : 0;

                        printf("  Measured:  %.3f ms/prediction  (%.1f iter/s, %d runs)\n",
                            avgMs, 1000.0/avgMs, nRuns);
                        if (grandGF > 0)
                            printf("  Compute:   %.2f GFLOP/s (%.4f TOPS)\n",
                                grandGF/avgMs*1000, grandGF/avgMs);
                        if (grandWeightMB > 0)
                            printf("  Weight BW: %.2f GB/s  (%.1f MB weights streamed/iter)\n",
                                grandWeightMB/avgMs, grandWeightMB);
                        printf("  Speedup:   %.1fx vs sequential estimate\n",
                            grandRT / avgMs);
                    }
                }
            } else {
                measurementStatus = @"unsupported_inputs";
                measurementError = @"Cannot auto-create dummy inputs (non-array inputs)";
                printf("  ⚠ Cannot auto-create dummy inputs (non-array inputs)\n");
            }
        }
    }

    // ── Top Expensive Ops ──────────────────────────────────────────
    {
        int topN = nUnique < 20 ? nUnique : 20;
        printf("\n═══════════════════════════════════════════════════════════════\n");
        printf("  Top %d Most Expensive Operations\n", topN);
        printf("═══════════════════════════════════════════════════════════════\n\n");

        printf("  %-44s %-24s %10s %8s %8s %s\n",
            "Op Name", "Type", "ms", "MB", "GB/s", "Bound");
        printf("  %-44s %-24s %10s %8s %8s %s\n",
            "────────────────────────────────────────────",
            "────────────────────────", "──────────", "────────", "────────", "──────");

        for (int i = 0; i < topN; i++) {
            CostEntry *e = &unique[i];
            double gbps = e->runtime > 0 ? e->totalMB / e->runtime : 0;
            char sname[45]; truncName(sname, e->name, 44);
            printf("  %-44s %-24s %10.6f %8.2f %8.2f %s\n",
                sname, shortType(e->type), e->runtime, e->totalMB, gbps,
                e->bound[0] ? e->bound : "?");
        }
    }

    // ── Conv detail ────────────────────────────────────────────────
    {
        int nConv = 0;
        for (int i = 0; i < nUnique; i++)
            if (strstr(unique[i].type, "conv")) nConv++;

        if (nConv > 0) {
            printf("\n═══════════════════════════════════════════════════════════════\n");
            printf("  Conv Detail (top 15)\n");
            printf("═══════════════════════════════════════════════════════════════\n\n");

            printf("  %-40s %10s %8s %8s %8s %6s %6s %7s\n",
                "Name", "ms", "GFLOP/s", "MB", "GB/s", "InCh", "OutCh", "WU%%");
            printf("  %-40s %10s %8s %8s %8s %6s %6s %7s\n",
                "────────────────────────────────────────",
                "──────────", "────────", "────────", "────────", "──────", "──────", "───────");

            int printed = 0;
            for (int i = 0; i < nUnique && printed < 15; i++) {
                CostEntry *e = &unique[i];
                if (!strstr(e->type, "conv")) continue;
                double gbps = e->runtime > 0 ? e->totalMB / e->runtime : 0;
                printf("  %-40s %10.6f %8.2f %8.2f %8.2f %6d %6d %6.1f%%\n",
                    e->name, e->runtime, e->gflops, e->totalMB, gbps,
                    e->inputCh, e->outputCh, e->workUnitEff * 100.0);
                printed++;
            }
        }
    }

    // ── High-cost MLComputePlan ops ────────────────────────────────
    if ([highCostOps count] > 0) {
        printf("\n═══════════════════════════════════════════════════════════════\n");
        printf("  MLComputePlan High-Cost Ops (weight > 0.5%%)\n");
        printf("═══════════════════════════════════════════════════════════════\n\n");

        printf("  %6s %-25s %8s %4s\n", "Index", "Op Type", "Cost", "Dev");
        printf("  %6s %-25s %8s %4s\n", "──────", "─────────────────────────",
            "────────", "────");
        for (NSDictionary *e in highCostOps)
            printf("  [%4d] %-25s %8.4f %s\n",
                [e[@"i"] intValue], [e[@"op"] UTF8String],
                [e[@"w"] doubleValue], [e[@"dev"] UTF8String]);
    }

    // ── ANE / CPU breakdown ────────────────────────────────────────
    if ([aneOpTypes count] > 0) {
        printf("\n═══════════════════════════════════════════════════════════════\n");
        printf("  ANE Op Types\n");
        printf("═══════════════════════════════════════════════════════════════\n\n");
        NSArray *s = [aneOpTypes keysSortedByValueUsingComparator:^(id a, id b) {
            return [b compare:a]; }];
        for (NSString *k in s)
            printf("  %-30s %4d\n", [k UTF8String], [aneOpTypes[k] intValue]);
    }
    if ([cpuOpTypes count] > 0) {
        printf("\n═══════════════════════════════════════════════════════════════\n");
        printf("  CPU Op Types\n");
        printf("═══════════════════════════════════════════════════════════════\n\n");
        NSArray *s = [cpuOpTypes keysSortedByValueUsingComparator:^(id a, id b) {
            return [b compare:a]; }];
        for (NSString *k in s)
            printf("  %-30s %4d\n", [k UTF8String], [cpuOpTypes[k] intValue]);
    }

    // ── CPU/GPU Fallback Detail ──────────────────────────────────
    if ([cpuOpDetails count] > 0) {
        printf("\n═══════════════════════════════════════════════════════════════════════════════════════════\n");
        printf("  CPU/GPU Fallback Ops (%d ops not on ANE)\n", (int)[cpuOpDetails count]);
        printf("═══════════════════════════════════════════════════════════════════════════════════════════\n\n");

        printf("  %-40s %-24s %4s %-12s %s\n",
            "Output Name", "Op Type", "Dev", "Supported", "Reason");
        printf("  %-40s %-24s %4s %-12s %s\n",
            "────────────────────────────────────────",
            "────────────────────────", "────", "────────────",
            "──────────────────────────────");

        // Sort by cost descending
        NSArray *sorted = [cpuOpDetails sortedArrayUsingComparator:
            ^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
                return [b[@"cost"] compare:a[@"cost"]];
            }];

        for (NSDictionary *d in sorted) {
            char sname[41]; truncName(sname, [d[@"name"] UTF8String], 40);
            // Look up ANE compiler reason by op type
            NSString *reason = d[@"reason"];
            NSMutableSet *reasons = unsupportedReasons[d[@"type"]];
            if (reasons && reasons.count > 0) {
                NSString *joined = [[reasons allObjects]
                    componentsJoinedByString:@"; "];
                reason = [NSString stringWithFormat:@"ane: %@", joined];
            }
            char sreason[81]; truncName(sreason, [reason UTF8String], 80);
            printf("  %-40s %-24s %4s %-12s %s\n",
                sname,
                shortType([d[@"type"] UTF8String]),
                [d[@"dev"] UTF8String],
                [d[@"supported"] UTF8String],
                sreason);
        }
    }

    printf("\n═══════════════════════════════════════════════════════════════\n");

write_json:
    if (jsonPath) {
        NSMutableDictionary *report = [NSMutableDictionary dictionary];
        report[@"tool"] = @"anemll-profile";
        report[@"version"] = [NSString stringWithUTF8String:VERSION];
        NSString *modelPath = [NSURL fileURLWithPath:modelcPath].path ?: modelcPath;
        report[@"model"] = @{
            @"input_path": modelPath ?: @"",
            @"compiled_model_path": modelcPath ?: @"",
            @"display_name": displayName ?: @"",
            @"format": isNeuralNet ? @"Neural Network" : @"ML Program",
            @"size_bytes": @(modelSize),
            @"size_mb": @(modelSize / 1048576.0),
            @"compiled_from_package": @(needsCompile)
        };
        if (resolvedFunctionName)
            report[@"function"] = resolvedFunctionName;
        report[@"compute"] = @{
            @"mode": [NSString stringWithUTF8String:unitsLabel],
            @"interrupt_boundary_ms": @(interruptBoundaryMs),
            @"compute_units_all": @(computeUnits == MLComputeUnitsAll)
        };
        report[@"status"] = @{
            @"no_costmodel_entries": @(noCostEntries),
            @"log_data_masked": @(logDataMasked)
        };
        report[@"summary"] = @{
            @"total_ops": @(totalOps),
            @"ane_ops": @(aneOps),
            @"cpu_ops": @(cpuOps),
            @"gpu_ops": @(gpuOps),
            @"const_ops": @(constOps),
            @"total_cost": @(totalCost),
            @"ane_cost": @(aneCost),
            @"cpu_cost": @(cpuCost),
            @"gpu_cost": @(gpuCost),
            @"costmodel_entries": @(nEntries),
            @"unique_costmodel_ops": @(nUnique),
            @"grand_runtime_ms": @(grandRT),
            @"grand_gflop": @(grandGF)
        };

        NSMutableDictionary *measurement = [NSMutableDictionary dictionary];
        measurement[@"status"] = measurementStatus ?: @"not_run";
        measurement[@"steady_ms"] = @(measuredSteadyMs);
        measurement[@"average_ms"] = @(measuredAvgMs);
        measurement[@"iter_per_sec"] = @(measuredIterPerSec);
        measurement[@"runs"] = @(measuredRuns);
        measurement[@"gflops_per_sec"] = @(measuredGFLOPs);
        measurement[@"tops"] = @(measuredTOPS);
        measurement[@"weight_bw_gbps"] = @(measuredWeightBW);
        measurement[@"speedup_vs_sequential"] = @(measuredSpeedup);
        if (measurementError) measurement[@"error"] = measurementError;
        report[@"measured_prediction"] = measurement;

        NSMutableArray *placementsJSON = [NSMutableArray array];
        for (int i = 0; i < nPlacements; i++) {
            PlacementEntry *p = &placements[i];
            if (p->isConst || !strcmp(p->device, "?")) continue;
            [placementsJSON addObject:@{
                @"function": [NSString stringWithUTF8String:p->function],
                @"index": @(p->index),
                @"name": [NSString stringWithUTF8String:p->name],
                @"type": [NSString stringWithUTF8String:p->type],
                @"short_type": [NSString stringWithUTF8String:shortType(p->type)],
                @"device": [NSString stringWithUTF8String:p->device],
                @"cost": @(p->cost),
                @"ane_supported": @(p->aneSupported)
            }];
        }
        report[@"ordered_placements"] = placementsJSON;

        NSMutableArray *interruptionsJSON = [NSMutableArray array];
        NSMutableArray *edgeRunsJSON = [NSMutableArray array];
        for (int i = 0; i < nIslands; i++) {
            DeviceIsland *d = &islands[i];
            char path[24], devs[16];
            runPathLabel(d, focusDevice, path, sizeof(path));
            runDeviceLabel(d, devs, sizeof(devs));
            NSMutableDictionary *item = [@{
                @"id": @(d->displayId),
                @"function": [NSString stringWithUTF8String:d->function],
                @"start_index": @(d->startIdx),
                @"end_index": @(d->endIdx),
                @"prev_focus_index": @(d->prevAneIdx),
                @"next_focus_index": @(d->nextAneIdx),
                @"function_total_ops": @(d->functionTotalOps),
                @"path": [NSString stringWithUTF8String:path],
                @"focus_device": [NSString stringWithUTF8String:focusDevice],
                @"device": [NSString stringWithUTF8String:devs],
                @"op_count": @(d->opCount),
                @"total_cost": @(d->totalCost),
                @"runtime_ms": @(d->runtimeMs),
                @"switch_penalty_ms": @(d->switchPenaltyMs),
                @"total_penalty_ms": @(d->totalPenaltyMs),
                @"soft_ops": @(d->softOps),
                @"hard_ops": @(d->hardOps),
                @"missing_runtime_ops": @(d->missingRuntimeOps),
                @"reason": d->mainReason[0] ? [NSString stringWithUTF8String:d->mainReason] : @"",
                @"first_type": [NSString stringWithUTF8String:d->firstType],
                @"last_type": [NSString stringWithUTF8String:d->lastType],
                @"first_short_type": [NSString stringWithUTF8String:shortType(d->firstType)],
                @"last_short_type": [NSString stringWithUTF8String:shortType(d->lastType)],
                @"center_percent": @(d->functionTotalOps > 0 ?
                    (((d->startIdx + d->endIdx) / 2.0) + 0.5) / d->functionTotalOps * 100.0 : 0.0),
                @"is_interruption": @(d->isInterruption)
            } mutableCopy];
            if (d->prevAneIdx < 0 && d->nextAneIdx < 0) item[@"edge_kind"] = @"all";
            else if (d->prevAneIdx < 0) item[@"edge_kind"] = @"lead";
            else if (d->nextAneIdx < 0) item[@"edge_kind"] = @"trail";
            if (d->isInterruption) [interruptionsJSON addObject:item];
            else [edgeRunsJSON addObject:item];
        }

        NSMutableArray *timelineJSON = [NSMutableArray array];
        for (int base = 0; base < nPlacements; ) {
            int end = base + 1;
            while (end < nPlacements &&
                   !strcmp(placements[end].function, placements[base].function)) end++;

            NSMutableArray *segments = [NSMutableArray array];
            for (int i = base; i < end; ) {
                while (i < end &&
                       (placements[i].isConst || !strcmp(placements[i].device, "?"))) i++;
                if (i >= end) break;

                PlacementEntry *p = &placements[i];
                int runStart = p->index;
                int runEnd = p->index;
                int j = i + 1;
                while (j < end) {
                    PlacementEntry *q = &placements[j];
                    if (q->isConst || !strcmp(q->device, "?")) { j++; continue; }
                    if (strcmp(q->device, p->device)) break;
                    runEnd = q->index;
                    j++;
                }

                NSMutableDictionary *seg = [@{
                    @"start_index": @(runStart),
                    @"end_index": @(runEnd),
                    @"device": [NSString stringWithUTF8String:p->device]
                } mutableCopy];
                for (int k = 0; k < nIslands; k++) {
                    DeviceIsland *d = &islands[k];
                    if (!strcmp(d->function, p->function) &&
                        d->startIdx == runStart && d->endIdx == runEnd) {
                        char matchDevs[16];
                        runDeviceLabel(d, matchDevs, sizeof(matchDevs));
                        seg[@"device"] = [NSString stringWithUTF8String:matchDevs];
                        if (d->isInterruption) {
                            seg[@"kind"] = @"interruption";
                            seg[@"interruption_id"] = @(d->displayId);
                        } else if (d->prevAneIdx < 0 && d->nextAneIdx < 0) {
                            seg[@"kind"] = @"all_non_ane";
                        } else if (d->prevAneIdx < 0) {
                            seg[@"kind"] = @"leading_non_ane";
                        } else {
                            seg[@"kind"] = @"trailing_non_ane";
                        }
                        break;
                    }
                }
                if (!seg[@"kind"]) {
                    seg[@"kind"] = !strcmp(p->device, focusDevice) ? @"focus" : @"non_focus";
                }
                [segments addObject:seg];
                i = j;
            }

            [timelineJSON addObject:@{
                @"function": [NSString stringWithUTF8String:placements[base].function],
                @"segments": segments
            }];
            base = end;
        }

        report[@"graph_interruptions"] = @{
            @"focus_device": [NSString stringWithUTF8String:focusDevice],
            @"non_focus_runs": @(nIslands),
            @"graph_interruptions": @(interruptionCount),
            @"edge_runs_ignored": @(edgeRunCount),
            @"interrupted_ops": @(interruptedOps),
            @"total_non_focus_ops": @(totalNonAneOps),
            @"interrupted_cost": @(interruptedCost),
            @"total_non_focus_cost": @(totalNonAneCost),
            @"estimated_switch_tax_ms": @(interruptedSwitchTaxMs),
            @"estimated_island_time_ms": @(interruptedRuntimeMs),
            @"estimated_total_tax_ms": @(interruptedPenaltyMs),
            @"best_case_savings_ms": @(interruptedPenaltyMs),
            @"interrupted_soft_ops": @(interruptedSoftOps),
            @"interrupted_hard_ops": @(interruptedHardOps),
            @"interrupted_missing_runtime_ops": @(interruptedMissingRuntimeOps),
            @"interruptions": interruptionsJSON,
            @"edge_runs": edgeRunsJSON,
            @"timeline_by_function": timelineJSON
        };

        NSMutableArray *typesJSON = [NSMutableArray array];
        for (int i = 0; i < nTypes; i++) {
            double pct = grandRT > 0 ? types[i].totalRuntime / grandRT * 100 : 0;
            double gbps = types[i].totalRuntime > 0 ? types[i].totalMB / types[i].totalRuntime : 0;
            const char *b = types[i].compBound > 0 ? "Comp" :
                (types[i].memBound > 0 ? "Mem" : "?");
            [typesJSON addObject:@{
                @"type": [NSString stringWithUTF8String:types[i].type],
                @"short_type": [NSString stringWithUTF8String:shortType(types[i].type)],
                @"count": @(types[i].count),
                @"ms_per_op": @(types[i].count > 0 ? types[i].totalRuntime / types[i].count : 0),
                @"total_ms": @(types[i].totalRuntime),
                @"gflop": @(types[i].totalGFlop),
                @"gbps": @(gbps),
                @"share_pct": @(pct),
                @"bound": [NSString stringWithUTF8String:b],
                @"weight_mb": @(types[i].weightMB)
            }];
        }
        report[@"op_type_breakdown"] = typesJSON;
        report[@"op_type_totals"] = @{
            @"grand_runtime_ms": @(grandRT),
            @"grand_gflop": @(grandGF),
            @"grand_weight_mb": @(grandWeightMB),
            @"grand_gbps": @(grandRT > 0 ? grandMB / grandRT : 0)
        };

        int topNj = nUnique < 20 ? nUnique : 20;
        NSMutableArray *topOpsJSON = [NSMutableArray array];
        NSMutableArray *convJSON = [NSMutableArray array];
        for (int i = 0; i < topNj; i++) {
            CostEntry *e = &unique[i];
            double gbps = e->runtime > 0 ? e->totalMB / e->runtime : 0;
            [topOpsJSON addObject:@{
                @"name": [NSString stringWithUTF8String:e->name],
                @"type": [NSString stringWithUTF8String:e->type],
                @"short_type": [NSString stringWithUTF8String:shortType(e->type)],
                @"runtime_ms": @(e->runtime),
                @"total_mb": @(e->totalMB),
                @"gbps": @(gbps),
                @"bound": e->bound[0] ? [NSString stringWithUTF8String:e->bound] : @"?"
            }];
        }
        for (int i = 0; i < nUnique && [convJSON count] < 15; i++) {
            CostEntry *e = &unique[i];
            if (!strstr(e->type, "conv")) continue;
            double gbps = e->runtime > 0 ? e->totalMB / e->runtime : 0;
            [convJSON addObject:@{
                @"name": [NSString stringWithUTF8String:e->name],
                @"type": [NSString stringWithUTF8String:e->type],
                @"runtime_ms": @(e->runtime),
                @"gflops": @(e->gflops),
                @"total_mb": @(e->totalMB),
                @"gbps": @(gbps),
                @"input_channels": @(e->inputCh),
                @"output_channels": @(e->outputCh),
                @"work_unit_efficiency_pct": @(e->workUnitEff * 100.0)
            }];
        }
        report[@"top_operations"] = topOpsJSON;
        report[@"conv_detail"] = convJSON;

        NSMutableArray *highCostJSON = [NSMutableArray array];
        for (NSDictionary *e in highCostOps) {
            [highCostJSON addObject:@{
                @"index": e[@"i"],
                @"type": e[@"op"],
                @"short_type": [NSString stringWithUTF8String:shortType([e[@"op"] UTF8String])],
                @"cost": e[@"w"],
                @"device": e[@"dev"]
            }];
        }
        report[@"high_cost_ops"] = highCostJSON;
        report[@"ane_op_types"] = [aneOpTypes copy];
        report[@"cpu_op_types"] = [cpuOpTypes copy];

        NSMutableDictionary *unsupportedJSON = [NSMutableDictionary dictionary];
        for (NSString *type in unsupportedReasons) {
            NSArray *reasons = [[unsupportedReasons[type] allObjects]
                sortedArrayUsingSelector:@selector(compare:)];
            unsupportedJSON[type] = reasons;
        }
        report[@"unsupported_reasons_by_type"] = unsupportedJSON;

        NSArray *sortedFallback = [cpuOpDetails sortedArrayUsingComparator:
            ^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
                return [b[@"cost"] compare:a[@"cost"]];
            }];
        NSMutableArray *fallbackJSON = [NSMutableArray array];
        for (NSDictionary *d in sortedFallback) {
            NSString *reason = d[@"reason"];
            NSMutableSet *reasons = unsupportedReasons[d[@"type"]];
            if (reasons && [reasons count] > 0) {
                NSString *joined = [[[reasons allObjects] sortedArrayUsingSelector:@selector(compare:)]
                    componentsJoinedByString:@"; "];
                reason = [NSString stringWithFormat:@"ane: %@", joined];
            }
            [fallbackJSON addObject:@{
                @"function": d[@"function"] ?: @"",
                @"index": d[@"idx"],
                @"name": d[@"name"],
                @"type": d[@"type"],
                @"short_type": [NSString stringWithUTF8String:shortType([d[@"type"] UTF8String])],
                @"device": d[@"dev"],
                @"cost": d[@"cost"],
                @"supported": d[@"supported"],
                @"reason": reason
            }];
        }
        report[@"fallback_ops"] = fallbackJSON;

        NSError *jsonErr = nil;
        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:report
            options:NSJSONWritingPrettyPrinted error:&jsonErr];
        if (!jsonData) {
            fprintf(stderr, "Error: could not serialize JSON report: %s\n",
                [[jsonErr localizedDescription] UTF8String]);
        } else {
            NSString *resolvedJSONPath = [jsonPath stringByStandardizingPath];
            NSString *jsonDir = [resolvedJSONPath stringByDeletingLastPathComponent];
            if ([jsonDir length] > 0 && ![fm fileExistsAtPath:jsonDir]) {
                NSError *mkdirErr = nil;
                [fm createDirectoryAtPath:jsonDir withIntermediateDirectories:YES
                    attributes:nil error:&mkdirErr];
                if (mkdirErr) {
                    fprintf(stderr, "Error: could not create JSON directory %s: %s\n",
                        [jsonDir UTF8String], [[mkdirErr localizedDescription] UTF8String]);
                    goto cleanup;
                }
            }
            NSError *writeErr = nil;
            if (![jsonData writeToFile:resolvedJSONPath options:NSDataWritingAtomic error:&writeErr]) {
                fprintf(stderr, "Error: could not write JSON report to %s: %s\n",
                    [resolvedJSONPath UTF8String], [[writeErr localizedDescription] UTF8String]);
            } else {
                printf("  JSON report: %s\n", [resolvedJSONPath UTF8String]);
            }
        }
    }

cleanup:
    if (g_logPath) { [fm removeItemAtPath:g_logPath error:nil]; g_logPath = nil; }
    free(entries);
    free(islands);
    free(placements);
    free(unique);
    return 0;
}

// ── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        BOOL showBanner = YES;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--no-banner")) { showBanner = NO; break; }
        }

        // os_log reads OS_ACTIVITY_DT_MODE during dyld init, before main() runs.
        // setenv() here is too late — re-exec so the variable is present from process start.
        if (!getenv("OS_ACTIVITY_DT_MODE")) {
            setenv("OS_ACTIVITY_DT_MODE", "YES", 1);
            char exepath[4096];
            uint32_t sz = sizeof(exepath);
            if (_NSGetExecutablePath(exepath, &sz) == 0)
                execv(exepath, argv);
            else
                execvp(argv[0], argv);
            // If re-exec fails, continue anyway
        }

        if (showBanner) {
            printf("anemll-profile %s\n", VERSION);
            printf("(C) 2026 ANEMLL (pronounced like \"animal\")\n");
            printf("Artificial Neural Engine Machine Learning Library, Open Source Project\n\n");
        }

        // Parse flags
        MLComputeUnits computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        const char *unitsLabel = "CPU+ANE";
        const char *modelArg = NULL;
        const char *compiledModelArg = NULL;
        double interruptBoundaryMs = 300.0;
        NSString *jsonPath = nil;
        NSString *requestedFunctionName = nil;
        NSString *displayNameOverride = nil;
        BOOL listFunctions = NO;
        BOOL allFunctions = NO;

        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
                printUsage(stdout);
                return 0;
            } else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--version")) {
                return 0;
            } else if (!strcmp(argv[i], "-a") || !strcmp(argv[i], "--all")) {
                computeUnits = MLComputeUnitsAll;
                unitsLabel = "All (CPU+GPU+ANE)";
            } else if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "--cpu-ane")) {
                computeUnits = MLComputeUnitsCPUAndNeuralEngine;
                unitsLabel = "CPU+ANE";
            } else if (!strcmp(argv[i], "--interrupt-boundary-ms") ||
                       !strcmp(argv[i], "--interrupt-ms")) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Error: %s requires a value in milliseconds\n", argv[i]);
                    return 1;
                }
                interruptBoundaryMs = atof(argv[++i]);
                if (interruptBoundaryMs < 0) {
                    fprintf(stderr, "Error: interruption boundary ms must be >= 0\n");
                    return 1;
                }
            } else if (!strcmp(argv[i], "--json") || !strcmp(argv[i], "-j")) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Error: %s requires an output file path\n", argv[i]);
                    return 1;
                }
                jsonPath = [NSString stringWithUTF8String:argv[++i]];
            } else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--function")) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Error: %s requires a function name.\n", argv[i]);
                    printUsage(stderr);
                    return 1;
                }
                requestedFunctionName = [NSString stringWithUTF8String:argv[++i]];
            } else if (!strcmp(argv[i], "--all-functions")) {
                allFunctions = YES;
            } else if (!strcmp(argv[i], "--list-functions")) {
                listFunctions = YES;
            } else if (!strcmp(argv[i], "--compiled-model")) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Error: %s requires a compiled model path.\n", argv[i]);
                    return 1;
                }
                compiledModelArg = argv[++i];
            } else if (!strcmp(argv[i], "--display-name")) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Error: %s requires a display name.\n", argv[i]);
                    return 1;
                }
                displayNameOverride = [NSString stringWithUTF8String:argv[++i]];
            } else if (!strcmp(argv[i], "--no-banner")) {
                continue;
            } else if (argv[i][0] == '-') {
                fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
                printUsage(stderr);
                return 1;
            } else if (!modelArg) {
                modelArg = argv[i];
            } else {
                fprintf(stderr, "Error: unexpected extra argument '%s'\n", argv[i]);
                printUsage(stderr);
                return 1;
            }
        }

        if ((requestedFunctionName != nil && allFunctions) ||
            (listFunctions && allFunctions) ||
            (listFunctions && requestedFunctionName != nil)) {
            fprintf(stderr, "Error: --function, --all-functions, and --list-functions are mutually exclusive selectors.\n");
            return 1;
        }

        if (!modelArg) {
            printUsage(stderr);
            return 1;
        }

        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *displayName = displayNameOverride;
        BOOL needsCompile = NO;
        NSString *modelPath = nil;
        if (compiledModelArg) {
            modelPath = [NSString stringWithUTF8String:compiledModelArg];
            if (!displayName) displayName = [modelPath lastPathComponent];
            if (![fm fileExistsAtPath:modelPath]) {
                fprintf(stderr, "Error: cannot find compiled model at '%s'\n", compiledModelArg);
                return 1;
            }
        } else {
            modelPath = resolveModelPath(modelArg, &displayName, &needsCompile);
            if (!modelPath) {
                fprintf(stderr, "Error: cannot find model at '%s'\n", modelArg);
                fprintf(stderr, "Tried: %s.mlmodelc, %s.mlpackage\n", modelArg, modelArg);
                return 1;
            }
        }

        NSString *modelcPath = modelPath;

        // ── Compile .mlpackage if needed ───────────────────────────────
        if (needsCompile) {
            printf("Compiling %s ...\n", [displayName UTF8String]);
            NSString *outDir = NSTemporaryDirectory();
            NSString *baseName = [[modelPath lastPathComponent] stringByDeletingPathExtension];
            NSString *cmd = [NSString stringWithFormat:
                @"xcrun coremlcompiler compile '%@' '%@' 2>&1", modelPath, outDir];
            int ret = system([cmd UTF8String]);
            if (ret != 0) {
                fprintf(stderr, "Error: coremlcompiler failed (exit %d)\n", ret);
                return 1;
            }
            modelcPath = [outDir stringByAppendingPathComponent:
                [baseName stringByAppendingString:@".mlmodelc"]];
            if (![fm fileExistsAtPath:modelcPath]) {
                fprintf(stderr, "Error: compiled model not found at %s\n", [modelcPath UTF8String]);
                return 1;
            }
            printf("Compiled OK\n\n");
        }

        // ── Model size ─────────────────────────────────────────────────
        unsigned long long modelSize = 0;
        NSDirectoryEnumerator *en = [fm enumeratorAtPath:modelcPath];
        NSString *file;
        while ((file = [en nextObject])) {
            NSDictionary *a = [fm attributesOfItemAtPath:
                [modelcPath stringByAppendingPathComponent:file] error:nil];
            modelSize += [a fileSize];
        }

        // ── MLModelAsset (macOS 15+) for multifunction support ─────────
        NSError *assetError = nil;
        MLModelAsset *modelAsset = nil;
        if (@available(macOS 15.0, *))
            modelAsset = copyModelAssetForCompiledModel(modelcPath, &assetError);

        // ── Function discovery for multifunction flags ─────────────────
        NSArray<NSString *> *assetFunctionNames = nil;
        BOOL needsFunctionDiscovery = listFunctions || allFunctions || requestedFunctionName != nil;
        if (needsFunctionDiscovery) {
            if (!supportsFunctionSelection()) {
                NSError *availabilityError = functionSelectionAvailabilityError();
                fprintf(stderr, "Error: %s\n",
                        [[availabilityError localizedDescription] UTF8String]);
                return 1;
            }
            NSError *functionError = nil;
            if (!modelAsset) {
                fprintf(stderr, "Error: %s\n",
                    assetError ? [[assetError localizedDescription] UTF8String]
                               : "Failed to create MLModelAsset");
                return 1;
            }
            if (!copyFunctionNamesForModelAsset(modelAsset, &assetFunctionNames, &functionError)) {
                fprintf(stderr, "Error: %s\n", [[functionError localizedDescription] UTF8String]);
                return 1;
            }

            BOOL singleFunctionModel = (assetFunctionNames.count == 0);
            NSArray<NSString *> *displayFunctionNames =
                singleFunctionModel ? @[@"main"] : assetFunctionNames;

            if (listFunctions) {
                for (NSString *name in displayFunctionNames)
                    printf("%s\n", [name UTF8String]);
                return 0;
            }

            if (singleFunctionModel) {
                if (requestedFunctionName && !isMainFunctionName(requestedFunctionName)) {
                    fprintf(stderr, "Error: single-function models only accept 'main'.\n");
                    return 1;
                }
                if (isMainFunctionName(requestedFunctionName))
                    requestedFunctionName = nil;
            } else if (requestedFunctionName &&
                       ![assetFunctionNames containsObject:requestedFunctionName]) {
                fprintf(stderr, "Error: function '%s' not found.\n",
                    [requestedFunctionName UTF8String]);
                fprintf(stderr, "Available functions:");
                for (NSString *name in assetFunctionNames)
                    fprintf(stderr, " %s", [name UTF8String]);
                fprintf(stderr, "\n");
                return 1;
            }

            if (allFunctions) {
                if (singleFunctionModel)
                    return profileModel(displayName, modelcPath, modelAsset,
                                        computeUnits, unitsLabel, modelArg,
                                        nil, interruptBoundaryMs, jsonPath,
                                        needsCompile, modelSize);

                // Multi-function: spawn a sub-process per function so each gets
                // a clean log capture session and separate Espresso cache clear.
                char exepath[4096];
                uint32_t exepathSize = sizeof(exepath);
                if (_NSGetExecutablePath(exepath, &exepathSize) != 0) {
                    fprintf(stderr, "Error: unable to resolve current executable path.\n");
                    return 1;
                }

                NSString *exePath = [NSString stringWithUTF8String:exepath];
                for (NSUInteger idx = 0; idx < assetFunctionNames.count; idx++) {
                    NSTask *task = [[NSTask alloc] init];
                    task.executableURL = [NSURL fileURLWithPath:exePath];
                    task.standardOutput = [NSFileHandle fileHandleWithStandardOutput];
                    task.standardError = [NSFileHandle fileHandleWithStandardError];

                    NSMutableArray<NSString *> *arguments = [NSMutableArray array];
                    [arguments addObject:@"--no-banner"];
                    if (computeUnits == MLComputeUnitsAll)
                        [arguments addObject:@"--all"];
                    if (interruptBoundaryMs != 300.0) {
                        [arguments addObject:@"--interrupt-boundary-ms"];
                        [arguments addObject:[NSString stringWithFormat:@"%.1f", interruptBoundaryMs]];
                    }
                    if (jsonPath) {
                        // For --all-functions with JSON, append function name to path
                        NSString *fnName = assetFunctionNames[idx];
                        NSString *base = [jsonPath stringByDeletingPathExtension];
                        NSString *ext = [jsonPath pathExtension];
                        NSString *fnJsonPath = ext.length > 0
                            ? [NSString stringWithFormat:@"%@.%@.%@", base, fnName, ext]
                            : [NSString stringWithFormat:@"%@.%@", base, fnName];
                        [arguments addObject:@"--json"];
                        [arguments addObject:fnJsonPath];
                    }
                    [arguments addObject:@"--compiled-model"];
                    [arguments addObject:modelcPath];
                    [arguments addObject:@"--display-name"];
                    [arguments addObject:displayName];
                    [arguments addObject:@"--function"];
                    [arguments addObject:assetFunctionNames[idx]];
                    [arguments addObject:[NSString stringWithUTF8String:modelArg]];
                    task.arguments = arguments;

                    NSError *launchError = nil;
                    if (![task launchAndReturnError:&launchError]) {
                        fprintf(stderr, "Error: %s\n", [[launchError localizedDescription] UTF8String]);
                        return 1;
                    }

                    if (idx > 0) printf("\n");
                    [task waitUntilExit];
                    if (task.terminationStatus != 0)
                        return task.terminationStatus;
                }
                return 0;
            }
        }

        return profileModel(displayName, modelcPath, modelAsset,
                            computeUnits, unitsLabel, modelArg,
                            requestedFunctionName, interruptBoundaryMs, jsonPath,
                            needsCompile, modelSize);
    }
}
