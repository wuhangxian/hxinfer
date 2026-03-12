#ifndef HXINFER_CONFIG_H
#define HXINFER_CONFIG_H
#include "cstdint"
namespace hxinfer{
    enum class DataType{
        kDataTypeUnknown=0,
        kDataTypeFP32=1,
        kDataTypeInt8=2,
    };

    inline int DataTypeSize(DataType data_type){
        if(data_type==DataType::kDataTypeFP32){
            return sizeof (float );
        }else if(data_type==DataType::kDataTypeInt8){
            return sizeof (int8_t);
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
    };
}

#endif //HXINFER_CONFIG_H
