#include "text_decoder.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>

bool read_npy_float(const char* filename, std::vector<float>& data, std::vector<int64_t>& shape) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    
    char magic[6];
    fread(magic, 1, 6, f);
    
    uint8_t major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);
    
    uint16_t header_len;
    fread(&header_len, 2, 1, f);
    
    std::vector<char> header(header_len + 1);
    fread(header.data(), 1, header_len, f);
    header[header_len] = 0;
    
    std::string hdr(header.data());
    size_t shape_start = hdr.find("'shape': (");
    if (shape_start == std::string::npos) {
        fclose(f);
        return false;
    }
    shape_start += 10;
    size_t shape_end = hdr.find(")", shape_start);
    std::string shape_str = hdr.substr(shape_start, shape_end - shape_start);
    
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        size_t comma = shape_str.find(',', pos);
        if (comma == std::string::npos) comma = shape_str.size();
        std::string num = shape_str.substr(pos, comma - pos);
        if (!num.empty() && num != " ") {
            shape.push_back(std::stoll(num));
        }
        pos = comma + 1;
    }
    
    int64_t total = 1;
    for (auto s : shape) total *= s;
    
    data.resize(total);
    fread(data.data(), sizeof(float), total, f);
    fclose(f);
    return true;
}

int main() {
    std::vector<float> hf_attn_weights, hf_attn_probs;
    std::vector<int64_t> shape_weights, shape_probs;
    
    if (!read_npy_float("/tmp/hf_attn_weights_10.npy", hf_attn_weights, shape_weights)) {
        fprintf(stderr, "Failed to load HF attention weights\n");
        return 1;
    }
    if (!read_npy_float("/tmp/hf_attn_probs_10.npy", hf_attn_probs, shape_probs)) {
        fprintf(stderr, "Failed to load HF attention probs\n");
        return 1;
    }
    
    printf("HF attention weights shape: [%lld, %lld, %lld, %lld]\n", 
           (long long)shape_weights[0], (long long)shape_weights[1], 
           (long long)shape_weights[2], (long long)shape_weights[3]);
    
    qwen3_asr::TextDecoder decoder;
    if (!decoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    if (!decoder.init_kv_cache(512)) {
        fprintf(stderr, "Failed to init KV cache: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    std::vector tokens = {151669, 151676, 151676, 151676, 151676,
                                    151676, 151676, 151676, 151676, 151670};
    
    printf("Running C++ decoder with %zu tokens...\n", tokens.size());
    
    std::vector<float> logits;
    std::map<std::string, std::vector<float>> debug_tensors;
    
    if (!decoder.forward_debug(tokens.data(), tokens.size(), 0, logits, debug_tensors)) {
        fprintf(stderr, "Forward pass failed: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    int n_heads = shape_weights[1];
    int seq_len = shape_weights[2];
    
    printf("\n=== HuggingFace attention weights (layer 0, head 0, 5x5) ===\n");
    for (int i = 0; i < 5 && i < seq_len; ++i) {
        for (int j = 0; j < 5 && j < seq_len; ++j) {
            int idx = 0 * n_heads * seq_len * seq_len + 0 * seq_len * seq_len + i * seq_len + j;
            printf("%8.4f ", hf_attn_weights[idx]);
        }
        printf("\n");
    }
    
    if (debug_tensors.count("debug_kq_scaled")) {
        const auto& cpp_scaled = debug_tensors["debug_kq_scaled"];
        printf("\n=== C++ KQ scaled (layer 0, head 0, 5x5) ===\n");
        printf("Total size: %zu\n", cpp_scaled.size());
        for (int i = 0; i < 5 && i < seq_len; ++i) {
            for (int j = 0; j < 5 && j < seq_len; ++j) {
                int idx = 0 * seq_len * seq_len + i * seq_len + j;
                if (idx < (int)cpp_scaled.size()) {
                    printf("%8.4f ", cpp_scaled[idx]);
                }
            }
            printf("\n");
        }
    }
    
    if (debug_tensors.count("debug_kq_softmax")) {
        const auto& cpp_probs = debug_tensors["debug_kq_softmax"];
        printf("\n=== C++ attention probs (layer 0, head 0, 5x5) ===\n");
        for (int i = 0; i < 5 && i < seq_len; ++i) {
            for (int j = 0; j < 5 && j < seq_len; ++j) {
                int idx = 0 * seq_len * seq_len + i * seq_len + j;
                if (idx < (int)cpp_probs.size()) {
                    printf("%8.4f ", cpp_probs[idx]);
                }
            }
            printf("\n");
        }
        
        printf("\n=== HuggingFace attention probs (layer 0, head 0, 5x5) ===\n");
        for (int i = 0; i < 5 && i < seq_len; ++i) {
            for (int j = 0; j < 5 && j < seq_len; ++j) {
                int idx = 0 * n_heads * seq_len * seq_len + 0 * seq_len * seq_len + i * seq_len + j;
                printf("%8.4f ", hf_attn_probs[idx]);
            }
            printf("\n");
        }
        
        printf("\n=== Difference (C++ - HF) ===\n");
        float max_diff = 0;
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                int cpp_idx = 0 * seq_len * seq_len + i * seq_len + j;
                int hf_idx = 0 * n_heads * seq_len * seq_len + 0 * seq_len * seq_len + i * seq_len + j;
                if (cpp_idx < (int)cpp_probs.size()) {
                    float diff = std::abs(cpp_probs[cpp_idx] - hf_attn_probs[hf_idx]);
                    if (diff > max_diff) max_diff = diff;
                }
            }
        }
        printf("Max difference in attention probs (head 0): %f\n", max_diff);
    }
    
    return 0;
}
