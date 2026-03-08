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
#define VERSION "0.3.1"

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

#define MAX_TYPES  128
#define MAX_ENTRIES 100000

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
    usleep(800000); // let log stream attach
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

// ── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        // os_log reads OS_ACTIVITY_DT_MODE during dyld init, before main() runs.
        // setenv() here is too late — re-exec so the variable is present from process start.
        // Use _NSGetExecutablePath for reliable re-exec (argv[0] may be a bare name
        // from PATH lookup, which fails with execv since it doesn't search PATH).
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

        printf("anemll-profile %s\n", VERSION);
        printf("(C) 2026 ANEMLL (pronounced like \"animal\")\n");
        printf("Artificial Neural Engine Machine Learning Library, Open Source Project\n\n");

        // Parse flags
        MLComputeUnits computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        const char *unitsLabel = "CPU+ANE";
        const char *modelArg = NULL;

        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--version")) {
                return 0;
            } else if (!strcmp(argv[i], "-a") || !strcmp(argv[i], "--all")) {
                computeUnits = MLComputeUnitsAll;
                unitsLabel = "All (CPU+GPU+ANE)";
            } else if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "--cpu-ane")) {
                computeUnits = MLComputeUnitsCPUAndNeuralEngine;
                unitsLabel = "CPU+ANE";
            } else if (argv[i][0] != '-') {
                modelArg = argv[i];
            }
        }

        if (!modelArg) {
            fprintf(stderr, "Usage: anemll-profile [options] <model_path>\n\n");
            fprintf(stderr, "  Profiles CoreML models via MLComputePlan + Espresso CostModel.\n");
            fprintf(stderr, "  Accepts .mlmodelc, .mlpackage, or base path (auto-detects).\n\n");
            fprintf(stderr, "  Options:\n");
            fprintf(stderr, "    -v, --version   Show version and exit\n");
            fprintf(stderr, "    -c, --cpu-ane   CPU + ANE (default)\n");
            fprintf(stderr, "    -a, --all       All devices incl. GPU\n\n");
            fprintf(stderr, "  Run: ./anemll-profile <model>\n");
            return 1;
        }

        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *displayName = nil;
        BOOL needsCompile = NO;
        NSString *modelPath = resolveModelPath(modelArg, &displayName, &needsCompile);

        if (!modelPath) {
            fprintf(stderr, "Error: cannot find model at '%s'\n", modelArg);
            fprintf(stderr, "Tried: %s.mlmodelc, %s.mlpackage\n", modelArg, modelArg);
            return 1;
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

        // ── Clear Espresso cache so CostModelFeature logs are emitted ──
        NSString *cacheDir = [NSHomeDirectory() stringByAppendingPathComponent:
            @"Library/Caches/anemll-profile/com.apple.e5rt.e5bundlecache"];
        if ([fm fileExistsAtPath:cacheDir])
            [fm removeItemAtPath:cacheDir error:nil];

        // ── Start log capture ──────────────────────────────────────────
        startLogCapture();

        // ── Load MLComputePlan ─────────────────────────────────────────
        NSURL *url = [NSURL fileURLWithPath:modelcPath];
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = computeUnits;

        __block BOOL done = NO;
        __block int totalOps = 0, aneOps = 0, cpuOps = 0, gpuOps = 0, constOps = 0;
        __block double totalCost = 0, aneCost = 0, cpuCost = 0;
        __block NSMutableDictionary *opTypeCounts = [NSMutableDictionary dictionary];
        __block NSMutableDictionary *aneOpTypes = [NSMutableDictionary dictionary];
        __block NSMutableDictionary *cpuOpTypes = [NSMutableDictionary dictionary];
        __block NSMutableArray *highCostOps = [NSMutableArray array];
        __block NSMutableArray *cpuOpDetails = [NSMutableArray array];
        __block BOOL isNeuralNet = NO;

        printf("Loading MLComputePlan ...\n");
        [MLComputePlan loadContentsOfURL:url configuration:config
            completionHandler:^(MLComputePlan *plan, NSError *error) {
                if (error) {
                    fprintf(stderr, "MLComputePlan error: %s\n",
                        [[error localizedDescription] UTF8String]);
                    done = YES;
                    return;
                }

                MLModelStructure *structure = [plan modelStructure];
                MLModelStructureProgram *prog = [structure program];

                if (prog) {
                    NSDictionary *funcs = [prog functions];
                    for (NSString *fname in funcs) {
                        MLModelStructureProgramFunction *fn = funcs[fname];
                        NSArray *ops = [[fn block] operations];
                        totalOps = (int)[ops count];

                        for (NSUInteger i = 0; i < [ops count]; i++) {
                            MLModelStructureProgramOperation *op = ops[i];
                            NSString *opName = [op operatorName];
                            MLComputePlanCost *cost = [plan estimatedCostOfMLProgramOperation:op];
                            MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:op];

                            double w = cost ? [cost weight] : 0;
                            if (!cost) constOps++;
                            totalCost += w;

                            opTypeCounts[opName] = @([opTypeCounts[opName] intValue] + 1);

                            NSString *devName = @"?";
                            BOOL aneSupported = NO;
                            if (usage) {
                                id pref = [usage preferredComputeDevice];
                                NSString *cls = NSStringFromClass([pref class]);
                                if ([cls containsString:@"NeuralEngine"]) {
                                    devName = @"ANE"; aneOps++; aneCost += w;
                                    aneOpTypes[opName] = @([aneOpTypes[opName] intValue] + 1);
                                } else if ([cls containsString:@"CPU"]) {
                                    devName = @"CPU"; cpuOps++; cpuCost += w;
                                    cpuOpTypes[opName] = @([cpuOpTypes[opName] intValue] + 1);
                                } else if ([cls containsString:@"GPU"]) {
                                    devName = @"GPU"; gpuOps++;
                                }
                                // Check if ANE is in supported devices
                                for (id dev in [usage supportedComputeDevices]) {
                                    NSString *dc = NSStringFromClass([dev class]);
                                    if ([dc containsString:@"NeuralEngine"]) {
                                        aneSupported = YES; break;
                                    }
                                }
                            }
                            // Collect non-ANE ops with reason
                            if (usage && ![devName isEqualToString:@"ANE"] && cost) {
                                // Get output name
                                NSString *outName = @"?";
                                NSArray *outputs = [op outputs];
                                if (outputs.count > 0) {
                                    outName = [outputs[0] name];
                                }
                                // Build supported list
                                NSMutableArray *supList = [NSMutableArray array];
                                for (id dev in [usage supportedComputeDevices]) {
                                    NSString *dc = NSStringFromClass([dev class]);
                                    if ([dc containsString:@"NeuralEngine"]) [supList addObject:@"ANE"];
                                    else if ([dc containsString:@"CPU"]) [supList addObject:@"CPU"];
                                    else if ([dc containsString:@"GPU"]) [supList addObject:@"GPU"];
                                }
                                NSString *reason = aneSupported ?
                                    @"ANE supported but not preferred" :
                                    @"Not supported on ANE";
                                [cpuOpDetails addObject:@{
                                    @"name": outName, @"type": opName,
                                    @"dev": devName, @"cost": @(w),
                                    @"supported": [supList componentsJoinedByString:@","],
                                    @"reason": reason,
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

                MLModelStructureNeuralNetwork *nn = [structure neuralNetwork];
                if (nn && !prog) {
                    isNeuralNet = YES;
                    NSArray *layers = [nn layers];
                    totalOps = (int)[layers count];
                    for (NSUInteger i = 0; i < [layers count]; i++) {
                        MLModelStructureNeuralNetworkLayer *layer = layers[i];
                        NSString *lt = [layer type];
                        opTypeCounts[lt] = @([opTypeCounts[lt] intValue] + 1);
                        MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForNeuralNetworkLayer:layer];
                        if (usage) {
                            NSString *cls = NSStringFromClass([[usage preferredComputeDevice] class]);
                            if ([cls containsString:@"NeuralEngine"]) {
                                aneOps++;
                                aneOpTypes[lt] = @([aneOpTypes[lt] intValue] + 1);
                            } else if ([cls containsString:@"CPU"]) {
                                cpuOps++;
                                cpuOpTypes[lt] = @([cpuOpTypes[lt] intValue] + 1);
                            } else if ([cls containsString:@"GPU"]) gpuOps++;
                        }
                    }
                }
                done = YES;
            }];

        NSDate *timeout = [NSDate dateWithTimeIntervalSinceNow:300];
        while (!done && [[NSDate date] compare:timeout] == NSOrderedAscending)
            [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode
                beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];

        sleep(1);
        stopLogCapture();

        if (!done) { fprintf(stderr, "Timeout\n"); return 1; }

        // ── Parse CostModelFeature logs ────────────────────────────────
        NSString *logContent = [NSString stringWithContentsOfFile:g_logPath
            encoding:NSUTF8StringEncoding error:nil];
        NSArray *logLines = [logContent componentsSeparatedByString:@"\n"];

        CostEntry *entries = calloc(MAX_ENTRIES, sizeof(CostEntry));
        int nEntries = 0;
        NSMutableSet *seenNames = [NSMutableSet set];
        CostEntry *unique = calloc(MAX_ENTRIES, sizeof(CostEntry));
        int nUnique = 0;

        // Parse "Unsupported op" reasons: "Unsupported op <idx> (<type>): <reason>"
        // Map type → set of reasons (most types have one consistent reason)
        NSMutableDictionary *unsupportedReasons = [NSMutableDictionary dictionary];

        for (NSString *line in logLines) {
            if (nEntries < MAX_ENTRIES) {
                CostEntry e;
                if (parseCostLine([line UTF8String], &e)) {
                    entries[nEntries++] = e;
                    NSString *ns = [NSString stringWithUTF8String:e.name];
                    if (![seenNames containsObject:ns]) {
                        [seenNames addObject:ns];
                        unique[nUnique++] = e;
                    }
                }
            }
            // Parse unsupported op reasons
            NSRange r = [line rangeOfString:@"Unsupported op "];
            if (r.location != NSNotFound) {
                NSString *rest = [line substringFromIndex:r.location + r.length];
                // Format: "<idx> (<type>): <reason>"
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
                            // Collect unique reasons per type
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
            // Weight DRAM traffic: mbKernel for conv/matmul, mbInput for LUT decompression
            if (strstr(e->type, "conv") || strstr(e->type, "matmul"))
                types[f].weightMB += e->mbKernel;
            else if (strstr(e->type, "constexpr_lut"))
                types[f].weightMB += e->mbInput;  // compressed weights are input
            // gather: only selected rows read, mbInput overestimates — skip
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
        printf("  Compute:      %s\n", unitsLabel);
        printf("  Total ops:    %d\n", totalOps);
        if (totalCost > 0) {
            printf("  ANE ops:      %d (%.1f%% of cost)\n", aneOps, aneCost/totalCost*100);
            printf("  CPU ops:      %d (%.1f%% of cost)\n", cpuOps, cpuCost/totalCost*100);
        } else {
            printf("  ANE ops:      %d\n", aneOps);
            printf("  CPU ops:      %d\n", cpuOps);
        }
        if (gpuOps) printf("  GPU ops:      %d\n", gpuOps);
        if (constOps) printf("  Const ops:    %d (no cost)\n", constOps);
        printf("  CostModel:    %d entries, %d unique ops\n", nEntries, nUnique);

        if (nUnique == 0) {
            printf("\n  ⚠ No CostModelFeature entries captured.\n");
            printf("  Try clearing cache: rm -rf ~/Library/Caches/anemll-profile/com.apple.e5rt*\n");
            goto cleanup;
        }

        // Detect <private> masking: many entries but only 1 unique means all names are "<private>"
        if (nUnique == 1 && nEntries > 10 && strstr(unique[0].name, "private")) {
            printf("\n  ⚠ Log data is masked (<private>). Run with:\n");
            printf("    OS_ACTIVITY_DT_MODE=YES anemll-profile %s\n", modelArg);
            goto cleanup;
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
            // Suppress stderr noise during model load (unsupported backend warnings)
            int saved_stderr = dup(STDERR_FILENO);
            int devnull = open("/dev/null", O_WRONLY);
            dup2(devnull, STDERR_FILENO);
            close(devnull);

            NSError *loadErr = nil;
            MLModelConfiguration *mcfg = [[MLModelConfiguration alloc] init];
            mcfg.computeUnits = computeUnits;
            MLModel *model = [MLModel modelWithContentsOfURL:url
                configuration:mcfg error:&loadErr];

            // Restore stderr
            dup2(saved_stderr, STDERR_FILENO);
            close(saved_stderr);
            if (loadErr) {
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
                        // State inputs handled separately via MLState
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

                    // Warmup: first run triggers compilation, second is steady-state
                    NSError *predErr = nil;
                    if (state)
                        [model predictionFromFeatures:provider
                            usingState:state error:&predErr];
                    else
                        [model predictionFromFeatures:provider error:&predErr];

                    if (predErr) {
                        printf("  ⚠ Prediction failed: %s\n",
                            [[predErr localizedDescription] UTF8String]);
                    } else {
                        // Second run = steady-state (compilation done)
                        uint64_t tw0 = mach_absolute_time();
                        predict();
                        uint64_t tw1 = mach_absolute_time();
                        double steadyMs = (double)(tw1 - tw0) * tbi.numer / tbi.denom / 1e6;

                        // Detect ANE compile failure: steady-state >> estimate
                        if (grandRT > 0 && steadyMs > grandRT * 3) {
                            printf("  ⚠ ANE compilation likely failed — model fell back to CPU\n");
                            printf("  CPU fallback: %.1f ms (vs %.1f ms estimated on ANE)\n",
                                steadyMs, grandRT);
                        } else {
                            // Adaptive run count
                            int nRuns = steadyMs > 1000 ? 1 : (steadyMs > 100 ? 3 : 10);

                            // Extra warmup for fast models
                            if (nRuns >= 10) predict();

                            // Timed runs
                            uint64_t t0 = mach_absolute_time();
                            for (int i = 0; i < nRuns; i++) predict();
                            uint64_t t1 = mach_absolute_time();

                            double totalNs = (double)(t1 - t0) * tbi.numer / tbi.denom;
                            double avgMs = totalNs / nRuns / 1e6;

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
                    printf("  ⚠ Cannot auto-create dummy inputs (non-array inputs)\n");
                }
            }
        }

        // ── Top Expensive Ops ──────────────────────────────────────────
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

        // ── Conv detail ────────────────────────────────────────────────
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

            int p = 0;
            for (int i = 0; i < nUnique && p < 15; i++) {
                CostEntry *e = &unique[i];
                if (!strstr(e->type, "conv")) continue;
                double gbps = e->runtime > 0 ? e->totalMB / e->runtime : 0;
                printf("  %-40s %10.6f %8.2f %8.2f %8.2f %6d %6d %6.1f%%\n",
                    e->name, e->runtime, e->gflops, e->totalMB, gbps,
                    e->inputCh, e->outputCh, e->workUnitEff * 100.0);
                p++;
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
                    // Join all unique reasons for this type
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

cleanup:
        if (g_logPath) [fm removeItemAtPath:g_logPath error:nil];
        free(entries);
        free(unique);
    }
    return 0;
}
