#include "text_decoder.h"

#include <cstdio>
#include <cstdlib>
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

static void compare_arrays(const char * name, const float * a, const float * b, size_t n) {
    float max_diff = 0.0f;
    size_t max_idx = 0;
    float sum_diff = 0.0f;
    
    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(a[i] - b[i]);
        sum_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    
    printf("%s: max_diff=%.6f at idx=%zu (cpp=%.6f, ref=%.6f), mean_diff=%.6f\n",
           name, max_diff, max_idx, a[max_idx], b[max_idx], sum_diff / n);
}

int main() {
    printf("=== Decoder Debug Test ===\n\n");
    
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
    printf("Reference embeddings shape: [%lld, %lld, %lld]\n", 
           (long long)shape[0], (long long)shape[1], (long long)shape[2]);
    printf("Reference embd[0,0,:10]: ");
    for (int i = 0; i < 10; ++i) printf("%.6f ", ref_embd[i]);
    printf("\n\n");
    
    if (!load_npy_f32("tests/reference/decoder_logits.npy", ref_logits, shape)) {
        fprintf(stderr, "Failed to load reference logits\n");
        return 1;
    }
    printf("Reference logits shape: [%lld, %lld, %lld]\n", 
           (long long)shape[0], (long long)shape[1], (long long)shape[2]);
    printf("Reference logits[0,0,:10]: ");
    for (int i = 0; i < 10; ++i) printf("%.6f ", ref_logits[i]);
    printf("\n\n");
    
    std::vector<float> logits;
    std::map<std::string, std::vector<float>> debug_tensors;
    if (!decoder.forward_debug(test_tokens.data(), test_tokens.size(), 0, logits, debug_tensors)) {
        fprintf(stderr, "Forward failed: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    printf("=== Debug Tensors ===\n");
    for (const auto & kv : debug_tensors) {
        printf("%s: size=%zu, first 10 values: ", kv.first.c_str(), kv.second.size());
        for (size_t i = 0; i < std::min(kv.second.size(), (size_t)10); ++i) {
            printf("%.6f ", kv.second[i]);
        }
        printf("\n");
    }
    printf("\n");
    
    std::vector<float> ref_norm0, ref_q0;
    std::vector<int64_t> norm0_shape, q0_shape;
    if (load_npy_f32("tests/reference/decoder_norm0.npy", ref_norm0, norm0_shape)) {
        printf("Reference norm0 shape: [%lld, %lld, %lld]\n", 
               (long long)norm0_shape[0], (long long)norm0_shape[1], (long long)norm0_shape[2]);
        printf("Reference norm0[0,0,:10]: ");
        for (int i = 0; i < 10; ++i) printf("%.6f ", ref_norm0[i]);
        printf("\n");
        
        if (debug_tensors.count("debug_norm0")) {
            compare_arrays("Norm0", debug_tensors["debug_norm0"].data(), ref_norm0.data(), 
                          std::min(debug_tensors["debug_norm0"].size(), ref_norm0.size()));
        }
    }
    
    if (load_npy_f32("tests/reference/decoder_q0.npy", ref_q0, q0_shape)) {
        printf("\nReference Q0 shape: [%lld, %lld, %lld]\n", 
               (long long)q0_shape[0], (long long)q0_shape[1], (long long)q0_shape[2]);
        printf("Reference Q0[0,0,:10]: ");
        for (int i = 0; i < 10; ++i) printf("%.6f ", ref_q0[i]);
        printf("\n");
        
        if (debug_tensors.count("debug_q0_raw")) {
            compare_arrays("Q0_raw", debug_tensors["debug_q0_raw"].data(), ref_q0.data(), 
                          std::min(debug_tensors["debug_q0_raw"].size(), ref_q0.size()));
        }
    }
    printf("\n");
    
    printf("C++ logits[0,:10]: ");
    for (int i = 0; i < 10; ++i) printf("%.6f ", logits[i]);
    printf("\n\n");
    
    compare_arrays("Logits (pos 0)", logits.data(), ref_logits.data(), cfg.vocab_size);
    
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
    
    printf("\nReference argmax: %d (logit=%.6f)\n", ref_argmax, ref_max);
    printf("C++ argmax: %d (logit=%.6f)\n", cpp_argmax, cpp_max);
    
    if (cpp_argmax == ref_argmax) {
        printf("\nTEST PASSED: Argmax matches!\n");
        return 0;
    } else {
        printf("\nTEST FAILED: Argmax mismatch!\n");
        return 1;
    }
}
