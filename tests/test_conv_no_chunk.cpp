#include "audio_encoder.h"

#include <cstdio>
#include <cmath>
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
    
    bool fortran_order = header_str.find("'fortran_order': True") != std::string::npos;
    
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
    
    if (fortran_order && shape.size() == 2) {
        std::vector<float> transposed(total_elements);
        int64_t rows = shape[0];
        int64_t cols = shape[1];
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                transposed[i * cols + j] = data[j * rows + i];
            }
        }
        data = std::move(transposed);
    }
    
    return true;
}

int main() {
    printf("=== Test Conv Layers (No Chunking) ===\n\n");
    
    // Load mel
    std::vector<float> mel_data;
    std::vector<int64_t> mel_shape;
    if (!load_npy_f32("tests/reference/mel.npy", mel_data, mel_shape)) {
        return 1;
    }
    printf("Mel shape: [%lld, %lld]\n", (long long)mel_shape[0], (long long)mel_shape[1]);
    
    // Load reference after_conv_out (no chunking)
    std::vector<float> ref_conv_out;
    std::vector<int64_t> ref_shape;
    if (!load_npy_f32("tests/reference/after_conv_out.npy", ref_conv_out, ref_shape)) {
        return 1;
    }
    printf("Reference after_conv_out shape: [%lld, %lld]\n", 
           (long long)ref_shape[0], (long long)ref_shape[1]);
    
    // Load model
    qwen3_asr::AudioEncoder encoder;
    if (!encoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", encoder.get_error().c_str());
        return 1;
    }
    printf("Model loaded\n\n");
    
    // Run encoder
    std::vector<float> output;
    int n_mel = mel_shape[0];
    int n_frames = mel_shape[1];
    
    if (!encoder.encode(mel_data.data(), n_mel, n_frames, output)) {
        fprintf(stderr, "Failed to encode: %s\n", encoder.get_error().c_str());
        return 1;
    }
    
    printf("Output size: %zu\n", output.size());
    printf("First 10 output values:\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] = %f\n", i, output[i]);
    }
    
    printf("\nFirst 10 reference values:\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] = %f\n", i, ref_conv_out[i]);
    }
    
    return 0;
}
