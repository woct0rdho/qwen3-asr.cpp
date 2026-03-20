#include "mel_spectrogram.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// Test configuration
constexpr float TOLERANCE = 1e-5f;
constexpr const char* DEFAULT_AUDIO_PATH = "sample.wav";
constexpr const char* DEFAULT_REFERENCE_PATH = "tests/reference/mel.npy";
constexpr const char* DEFAULT_FILTERS_PATH = "tests/reference/mel_filters.npy";
constexpr const char* DEFAULT_OUTPUT_PATH = "tests/output/mel_computed.npy";

void print_usage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  --audio <path>      Input audio file (default: %s)\n", DEFAULT_AUDIO_PATH);
    printf("  --reference <path>  Reference mel.npy file (default: %s)\n", DEFAULT_REFERENCE_PATH);
    printf("  --filters <path>    Mel filters .npy file (default: %s)\n", DEFAULT_FILTERS_PATH);
    printf("  --output <path>     Output mel.npy file (default: %s)\n", DEFAULT_OUTPUT_PATH);
    printf("  --tolerance <val>   Comparison tolerance (default: %.0e)\n", TOLERANCE);
    printf("  --threads <n>       Number of threads (default: 1)\n");
    printf("  --help              Show this help message\n");
}

int main(int argc, char** argv) {
    std::string audio_path = DEFAULT_AUDIO_PATH;
    std::string reference_path = DEFAULT_REFERENCE_PATH;
    std::string filters_path = DEFAULT_FILTERS_PATH;
    std::string output_path = DEFAULT_OUTPUT_PATH;
    float tolerance = TOLERANCE;
    int n_threads = 1;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--audio" && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (arg == "--reference" && i + 1 < argc) {
            reference_path = argv[++i];
        } else if (arg == "--filters" && i + 1 < argc) {
            filters_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--tolerance" && i + 1 < argc) {
            tolerance = std::stof(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            n_threads = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("=== Mel Spectrogram Test ===\n");
    printf("Audio:     %s\n", audio_path.c_str());
    printf("Reference: %s\n", reference_path.c_str());
    printf("Filters:   %s\n", filters_path.c_str());
    printf("Output:    %s\n", output_path.c_str());
    printf("Tolerance: %.0e\n", tolerance);
    printf("Threads:   %d\n", n_threads);
    printf("\n");

    // Step 1: Load audio
    printf("Loading audio...\n");
    std::vector<float> samples;
    int sample_rate;
    if (!load_wav(audio_path, samples, sample_rate)) {
        fprintf(stderr, "FAILED: Could not load audio file\n");
        return 1;
    }
    printf("  Loaded %zu samples at %d Hz (%.2f seconds)\n", 
           samples.size(), sample_rate, static_cast<float>(samples.size()) / sample_rate);

    if (sample_rate != QWEN_SAMPLE_RATE) {
        fprintf(stderr, "WARNING: Sample rate is %d Hz, expected %d Hz\n", 
                sample_rate, QWEN_SAMPLE_RATE);
    }

    // Step 2: Load mel filterbank
    printf("Loading mel filterbank...\n");
    MelFilters filters;
    if (!load_mel_filters_npy(filters_path, filters)) {
        fprintf(stderr, "FAILED: Could not load mel filters\n");
        return 1;
    }
    printf("  Loaded filters: n_mel=%d, n_fft=%d\n", filters.n_mel, filters.n_fft);

    // Step 3: Compute mel spectrogram
    printf("Computing mel spectrogram...\n");
    MelSpectrogram mel_computed;
    if (!log_mel_spectrogram(samples.data(), static_cast<int>(samples.size()), 
                             filters, mel_computed, n_threads)) {
        fprintf(stderr, "FAILED: Could not compute mel spectrogram\n");
        return 1;
    }
    printf("  Computed mel: n_mel=%d, n_len=%d, n_len_org=%d\n", 
           mel_computed.n_mel, mel_computed.n_len, mel_computed.n_len_org);

    // Step 4: Save computed mel spectrogram
    printf("Saving computed mel spectrogram...\n");
    // Create output directory if needed
    std::string output_dir = output_path.substr(0, output_path.find_last_of('/'));
    if (!output_dir.empty()) {
        std::string mkdir_cmd = "mkdir -p " + output_dir;
        system(mkdir_cmd.c_str());
    }
    if (!save_mel_npy(output_path, mel_computed)) {
        fprintf(stderr, "WARNING: Could not save computed mel spectrogram\n");
    } else {
        printf("  Saved to: %s\n", output_path.c_str());
    }

    // Step 5: Load reference mel spectrogram
    printf("Loading reference mel spectrogram...\n");
    MelSpectrogram mel_reference;
    if (!load_mel_npy(reference_path, mel_reference)) {
        fprintf(stderr, "FAILED: Could not load reference mel spectrogram\n");
        return 1;
    }
    printf("  Reference mel: n_mel=%d, n_len=%d\n", 
           mel_reference.n_mel, mel_reference.n_len);

    // Step 6: Compare
    printf("Comparing mel spectrograms...\n");
    
    // Check dimensions
    if (mel_computed.n_mel != mel_reference.n_mel) {
        fprintf(stderr, "FAILED: n_mel mismatch: %d vs %d\n", 
                mel_computed.n_mel, mel_reference.n_mel);
        return 1;
    }

    // For comparison, we only compare up to the reference length
    // (computed may have more frames due to padding)
    int compare_len = std::min(mel_computed.n_len, mel_reference.n_len);
    printf("  Comparing %d x %d values\n", mel_computed.n_mel, compare_len);

    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int num_values = mel_computed.n_mel * compare_len;
    int num_above_tolerance = 0;

    for (int j = 0; j < mel_computed.n_mel; j++) {
        for (int i = 0; i < compare_len; i++) {
            float computed = mel_computed.data[j * mel_computed.n_len + i];
            float reference = mel_reference.data[j * mel_reference.n_len + i];
            float diff = std::abs(computed - reference);
            
            sum_diff += diff;
            if (diff > max_diff) {
                max_diff = diff;
            }
            if (diff > tolerance) {
                num_above_tolerance++;
            }
        }
    }

    float mean_diff = sum_diff / num_values;

    printf("\n=== Results ===\n");
    printf("Max absolute difference:  %.6e\n", max_diff);
    printf("Mean absolute difference: %.6e\n", mean_diff);
    printf("Values above tolerance:   %d / %d (%.2f%%)\n", 
           num_above_tolerance, num_values, 
           100.0f * num_above_tolerance / num_values);

    // Print some sample values for debugging
    printf("\nSample values (first 5 frames, first 5 mel bins):\n");
    printf("%-10s %-12s %-12s %-12s\n", "Position", "Computed", "Reference", "Diff");
    for (int j = 0; j < std::min(5, mel_computed.n_mel); j++) {
        for (int i = 0; i < std::min(5, compare_len); i++) {
            float computed = mel_computed.data[j * mel_computed.n_len + i];
            float reference = mel_reference.data[j * mel_reference.n_len + i];
            float diff = std::abs(computed - reference);
            printf("[%d,%d]     %-12.6f %-12.6f %-12.6e\n", 
                   j, i, computed, reference, diff);
        }
    }

    // Final verdict
    printf("\n");
    if (max_diff <= tolerance) {
        printf("PASSED: Max difference %.6e <= tolerance %.6e\n", max_diff, tolerance);
        return 0;
    } else {
        printf("FAILED: Max difference %.6e > tolerance %.6e\n", max_diff, tolerance);
        return 1;
    }
}
