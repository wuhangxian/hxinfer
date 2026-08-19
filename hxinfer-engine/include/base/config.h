#ifndef HXINFER_CONFIG_H
#define HXINFER_CONFIG_H
#include "cstdint"
namespace hxinfer{
    enum class DataType{
        kDataTypeUnknown=0,
        kDataTypeFP32=1,
        kDataTypeInt8=2,
        kDataTypeFP16=3,
    };

    enum class DeviceType{
        kDeviceCPU=0,
        kDeviceCUDA=1
    };

    inline int DataTypeSize(DataType data_type){
        if(data_type==DataType::kDataTypeFP32){
            return sizeof (float );
        }else if(data_type==DataType::kDataTypeInt8){
            return sizeof (int8_t);
        }else if(data_type==DataType::kDataTypeFP16){
            return sizeof (uint16_t);
        }else{
            return 0;
        }
    }
    struct ModelConfig{
        int dim;
        int hidden_dim;
        int layer;
        int head;
        int kv_head;
        int vocab_size;
        int seq_len;

        // YaRN RoPE scaling parameters
        float rope_factor       = 1.0f;   // scaling factor (e.g. 32.0 for YaRN-128k)
        float rope_beta_fast    = 32.0f;  // default in HuggingFace YaRN
        float rope_beta_slow    = 1.0f;   // default in HuggingFace YaRN
        float rope_theta       = 10000.0f; // RoPE base frequency
        int   rope_orig_max_pos = 4096;   // original_max_position_embeddings
        bool  rope_use_yarn     = false;   // true when type=="yarn"

        // Precision config (centralized — all layers read from here)
        DataType weight_dtype      = DataType::kDataTypeFP16;  // model weights
        DataType activation_dtype  = DataType::kDataTypeFP32;  // hidden states, QKV, KV cache
        DataType norm_weight_dtype = DataType::kDataTypeFP32;  // RMSNorm weights (small, FP32 for precision)
        DataType bias_dtype        = DataType::kDataTypeFP32;  // QKV bias (must match activation_dtype)
        DataType logits_dtype      = DataType::kDataTypeFP32;  // final logits output
    };
}

#endif //HXINFER_CONFIG_H
