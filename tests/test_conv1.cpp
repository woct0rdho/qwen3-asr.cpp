#include "audio_encoder.h"

#include <cstdio>
#include <vector>
#include <string>

static bool load_npy_f32(const std::string & path, std::vector<float> & data, 
                         std::vector<int64_t> & shape) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) { fclose(f); return false; }
    if (magic[0] != '\x93') { fclose(f); return false; }
    
    uint8_t major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);
    
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16;
        fread(&len16, 2, 1, f);
        header_len = len16;
    } else {
        fread(&header_len, 4, 1, f);
    }
    
    std::vector<char> header(header_len + 1);
    fread(header.data(), 1, header_len, f);
    header[header_len] = '\0';
    
    std::string header_str(header.data());
    size_t shape_start = header_str.find("'shape': (") + 10;
    size_t shape_end = header_str.find(")", shape_start);
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
    fread(data.data(), sizeof(float), total, f);
    fclose(f);
    return true;
}

int main() {
    printf("=== Test Conv1 Output ===\n\n");
    
    // Load mel
    std::vector<float> mel_data;
    std::vector<int64_t> mel_shape;
    load_npy_f32("tests/reference/mel.npy", mel_data, mel_shape);
    printf("Mel shape: [%lld, %lld]\n", (long long)mel_shape[0], (long long)mel_shape[1]);
    
    // Load conv1 reference
    std::vector<float> conv1_ref;
    std::vector<int64_t> conv1_shape;
    load_npy_f32("tests/reference/conv1_out.npy", conv1_ref, conv1_shape);
    printf("Conv1 ref shape: [%lld, %lld, %lld, %lld]\n", 
           (long long)conv1_shape[0], (long long)conv1_shape[1], 
           (long long)conv1_shape[2], (long long)conv1_shape[3]);
    
    // Load model
    qwen3_asr::AudioEncoder encoder;
    if (!encoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    // We need to access the internal state to run just conv1
    // For now, let's just check the kernel values
    
    printf("\nChecking kernel values...\n");
    
    // The encoder doesn't expose the kernel directly, so let's check via the GGUF file
    // Actually, let's just run the full conv and check the output
    
    std::vector<float> output;
    int n_mel = mel_shape[0];
    int n_frames = mel_shape[1];
    
    if (!encoder.encode_conv_only(mel_data.data(), n_mel, n_frames, output)) {
        fprintf(stderr, "Failed to encode\n");
        return 1;
    }
    
    printf("Output size: %zu\n", output.size());
    
    // The conv1 reference is [1, 480, 64, 1500] in HuggingFace format
    // After GELU, the values should match
    
    // Let's check the first few values of the final conv output
    printf("\nFirst 20 output values:\n");
    for (int i = 0; i < 20; i++) {
        printf("  [%d] = %f\n", i, output[i]);
    }
    
    // Load after_conv_out reference
    std::vector<float> ref_data;
    std::vector<int64_t> ref_shape;
    load_npy_f32("tests/reference/after_conv_out.npy", ref_data, ref_shape);
    
    printf("\nFirst 20 reference values:\n");
    for (int i = 0; i < 20; i++) {
        printf("  [%d] = %f\n", i, ref_data[i]);
    }
    
    return 0;
}
