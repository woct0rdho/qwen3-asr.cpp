#include "audio_encoder.h"

#include <cstdio>
#include <cstdlib>

static bool load_npy_f32(const std::string & path, std::vector<float> & data, 
                         std::vector<int64_t> & shape) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) { fclose(f); return false; }
    if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' || 
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        fclose(f); return false;
    }
    
    uint8_t major, minor;
    if (fread(&major, 1, 1, f) != 1 || fread(&minor, 1, 1, f) != 1) {
        fclose(f); return false;
    }
    
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16;
        if (fread(&len16, 2, 1, f) != 1) { fclose(f); return false; }
        header_len = len16;
    } else {
        if (fread(&header_len, 4, 1, f) != 1) { fclose(f); return false; }
    }
    
    std::vector<char> header(header_len + 1);
    if (fread(header.data(), 1, header_len, f) != header_len) {
        fclose(f); return false;
    }
    header[header_len] = '\0';
    
    std::string header_str(header.data());
    
    size_t shape_start = header_str.find("'shape': (");
    if (shape_start == std::string::npos) { fclose(f); return false; }
    shape_start += 10;
    
    size_t shape_end = header_str.find(")", shape_start);
    if (shape_end == std::string::npos) { fclose(f); return false; }
    
    std::string shape_str = header_str.substr(shape_start, shape_end - shape_start);
    
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
        if (pos >= shape_str.size()) break;
        int64_t dim = 0;
        while (pos < shape_str.size() && shape_str[pos] >= '0' && shape_str[pos] <= '9') {
            dim = dim * 10 + (shape_str[pos] - '0');
            pos++;
        }
        shape.push_back(dim);
    }
    
    int64_t total = 1;
    for (auto d : shape) total *= d;
    
    data.resize(total);
    if (fread(data.data(), sizeof(float), total, f) != (size_t)total) {
        fclose(f); return false;
    }
    
    fclose(f);
    return true;
}

int main() {
    printf("=== Test Conv Only (No Chunking) ===\n\n");
    
    std::vector<float> mel_data;
    std::vector<int64_t> mel_shape;
    if (!load_npy_f32("tests/reference/mel.npy", mel_data, mel_shape)) {
        fprintf(stderr, "Failed to load mel\n");
        return 1;
    }
    printf("Mel shape: [%lld, %lld]\n", (long long)mel_shape[0], (long long)mel_shape[1]);
    
    std::vector<float> ref_data;
    std::vector<int64_t> ref_shape;
    if (!load_npy_f32("tests/reference/after_conv_out.npy", ref_data, ref_shape)) {
        fprintf(stderr, "Failed to load reference\n");
        return 1;
    }
    
    if (ref_shape.size() == 3 && ref_shape[0] == 1) {
        ref_shape.erase(ref_shape.begin());
    }
    printf("Reference shape: [%lld, %lld]\n", (long long)ref_shape[0], (long long)ref_shape[1]);
    
    qwen3_asr::AudioEncoder encoder;
    if (!encoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", encoder.get_error().c_str());
        return 1;
    }
    printf("Model loaded\n\n");
    
    std::vector<float> output;
    int n_mel = mel_shape[0];
    int n_frames = mel_shape[1];
    
    if (!encoder.encode_conv_only(mel_data.data(), n_mel, n_frames, output)) {
        fprintf(stderr, "Failed to encode: %s\n", encoder.get_error().c_str());
        return 1;
    }
    
    int64_t out_seq = output.size() / 896;
    printf("Output shape: [%lld, 896]\n", (long long)out_seq);
    printf("Reference shape: [%lld, %lld]\n", (long long)ref_shape[0], (long long)ref_shape[1]);
    
    if (output.size() != ref_data.size()) {
        fprintf(stderr, "Size mismatch: got %zu, expected %zu\n", output.size(), ref_data.size());
        
        printf("\nFirst 10 output values:\n");
        for (int i = 0; i < 10 && i < (int)output.size(); i++) {
            printf("  [%d] = %f\n", i, output[i]);
        }
        
        printf("\nFirst 10 reference values:\n");
        for (int i = 0; i < 10 && i < (int)ref_data.size(); i++) {
            printf("  [%d] = %f\n", i, ref_data[i]);
        }
        return 1;
    }
    
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    for (size_t i = 0; i < output.size(); i++) {
        float diff = std::abs(output[i] - ref_data[i]);
        max_diff = std::max(max_diff, diff);
        mean_diff += diff;
    }
    mean_diff /= output.size();
    
    printf("\nComparison results:\n");
    printf("  Max diff: %e\n", max_diff);
    printf("  Mean diff: %e\n", mean_diff);
    
    if (max_diff < 1e-3) {
        printf("\nTEST PASSED\n");
        return 0;
    } else {
        printf("\nTEST FAILED\n");
        
        printf("\nFirst 10 differences:\n");
        int count = 0;
        for (size_t i = 0; i < output.size() && count < 10; i++) {
            float diff = std::abs(output[i] - ref_data[i]);
            if (diff > 1e-3) {
                printf("  [%zu] output=%f, ref=%f, diff=%e\n", i, output[i], ref_data[i], diff);
                count++;
            }
        }
        return 1;
    }
}
