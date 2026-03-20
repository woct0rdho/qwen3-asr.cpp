#include "text_decoder.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>

static bool load_npy_f32(const std::string & path, std::vector<float> & data, 
                         std::vector<int64_t> & shape) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", path.c_str());
        return false;
    }
    
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) {
        fclose(f);
        return false;
    }
    
    if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' || 
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        fprintf(stderr, "Invalid NPY magic: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    
    uint8_t major, minor;
    if (fread(&major, 1, 1, f) != 1 || fread(&minor, 1, 1, f) != 1) {
        fclose(f);
        return false;
    }
    
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16;
        if (fread(&len16, 2, 1, f) != 1) {
            fclose(f);
            return false;
        }
        header_len = len16;
    } else {
        if (fread(&header_len, 4, 1, f) != 1) {
            fclose(f);
            return false;
        }
    }
    
    std::vector<char> header(header_len + 1);
    if (fread(header.data(), 1, header_len, f) != header_len) {
        fclose(f);
        return false;
    }
    header[header_len] = '\0';
    
    std::string header_str(header.data());
    
    size_t shape_start = header_str.find("'shape': (");
    if (shape_start == std::string::npos) {
        fprintf(stderr, "Failed to find shape in header: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    shape_start += 10;
    
    size_t shape_end = header_str.find(")", shape_start);
    if (shape_end == std::string::npos) {
        fclose(f);
        return false;
    }
    
    std::string shape_str = header_str.substr(shape_start, shape_end - shape_start);
    
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) {
            pos++;
        }
        if (pos >= shape_str.size()) break;
        
        int64_t dim = 0;
        while (pos < shape_str.size() && shape_str[pos] >= '0' && shape_str[pos] <= '9') {
            dim = dim * 10 + (shape_str[pos] - '0');
            pos++;
        }
        shape.push_back(dim);
    }
    
    int64_t total_elements = 1;
    for (auto d : shape) {
        total_elements *= d;
    }
    
    data.resize(total_elements);
    
    if (fread(data.data(), sizeof(float), total_elements, f) != (size_t)total_elements) {
        fprintf(stderr, "Failed to read data from: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    
    fclose(f);
    return true;
}

int main() {
    printf("=== Decoder Trace Test ===\n\n");
    
    qwen3_asr::TextDecoder decoder;
    
    printf("Loading model...\n");
    if (!decoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    const auto & cfg = decoder.get_config();
    printf("Model config:\n");
    printf("  vocab_size: %d\n", cfg.vocab_size);
    printf("  hidden_size: %d\n", cfg.hidden_size);
    printf("  n_decoder_layers: %d\n", cfg.n_decoder_layers);
    printf("  n_attention_heads: %d\n", cfg.n_attention_heads);
    printf("  n_key_value_heads: %d\n", cfg.n_key_value_heads);
    printf("  head_dim: %d\n", cfg.head_dim);
    printf("\n");
    
    if (!decoder.init_kv_cache(512)) {
        fprintf(stderr, "Failed to init KV cache: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    std::vector<int32_t> test_tokens = {151669, 151676, 151676, 151676, 151670};
    
    printf("Test tokens: ");
    for (auto t : test_tokens) printf("%d ", t);
    printf("\n\n");
    
    std::vector<float> ref_embd, ref_logits;
    std::vector<int64_t> shape;
    
    if (!load_npy_f32("tests/reference/decoder_embd.npy", ref_embd, shape)) {
        fprintf(stderr, "Failed to load reference embeddings\n");
        return 1;
    }
    printf("Reference embeddings [0,0,:10]: ");
    for (int i = 0; i < 10; ++i) printf("%.6f ", ref_embd[i]);
    printf("\n");
    
    if (!load_npy_f32("tests/reference/decoder_logits.npy", ref_logits, shape)) {
        fprintf(stderr, "Failed to load reference logits\n");
        return 1;
    }
    printf("Reference logits [0,0,:10]: ");
    for (int i = 0; i < 10; ++i) printf("%.6f ", ref_logits[i]);
    printf("\n\n");
    
    std::vector<float> logits;
    if (!decoder.forward(test_tokens.data(), test_tokens.size(), 0, logits)) {
        fprintf(stderr, "Forward failed: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    printf("C++ logits [0,:10]: ");
    for (int i = 0; i < 10; ++i) printf("%.6f ", logits[i]);
    printf("\n\n");
    
    int ref_argmax = 0;
    float ref_max = ref_logits[0];
    for (int i = 1; i < cfg.vocab_size; ++i) {
        if (ref_logits[i] > ref_max) {
            ref_max = ref_logits[i];
            ref_argmax = i;
        }
    }
    
    int cpp_argmax = 0;
    float cpp_max = logits[0];
    for (int i = 1; i < cfg.vocab_size; ++i) {
        if (logits[i] > cpp_max) {
            cpp_max = logits[i];
            cpp_argmax = i;
        }
    }
    
    printf("Reference argmax: %d (logit=%.6f)\n", ref_argmax, ref_max);
    printf("C++ argmax: %d (logit=%.6f)\n", cpp_argmax, cpp_max);
    
    float max_diff = 0.0f;
    for (int i = 0; i < cfg.vocab_size; ++i) {
        float diff = std::abs(logits[i] - ref_logits[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max logit diff: %.6f\n", max_diff);
    
    if (max_diff < 0.1f) {
        printf("\nTEST PASSED: Logits match within tolerance!\n");
        return 0;
    } else {
        printf("\nTEST FAILED: Logits differ significantly!\n");
        return 1;
    }
}
