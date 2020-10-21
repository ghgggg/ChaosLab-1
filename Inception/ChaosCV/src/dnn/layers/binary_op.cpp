#include "dnn/layers/binary_op.hpp"

namespace chaos
{
	namespace dnn
	{
        struct BinaryAdd
        {
            float operator()(const float& x, const float& y) const { return x + y; }
        };

        struct BinarySub
        {
            float operator()(const float& x, const float& y) const { return x - y; }
        };

        struct BinaryMul
        {
            float operator()(const float& x, const float& y) const { return x * y; }
        };

        struct BinaryDiv
        {
            float operator()(const float& x, const float& y) const { return x / y; }
        };

        template<typename Op>
        static void Operator(const Tensor& a, const Tensor& b, Tensor& c, const Option& opt)
        {
            Op op;

            if (a.shape == b.shape)
            {
                size_t n = a.shape.vol(); // (size_t)a.shape[0] * a.steps[0] / (1 * a.depth);
                c.Create(a.shape, a.depth, a.packing, opt.blob_allocator);

                for (size_t i = 0; i < n; i++)
                {
                    c[i] = op(a[i], b[i]);
                }
            }
            else
            {
                // CHECK_EQ(a.shape.size(), b.shape.size());
                CHECK_EQ(a.shape.size(), b.shape.size());
                auto shape = a.shape.vol() > b.shape.vol() ? a.shape : b.shape;

                c.Create(shape, a.depth, a.packing, opt.blob_allocator);

                size_t n = shape.vol();
                size_t num_axes = shape.size();
                for (size_t i = 0; i < n; i++)
                {
                    size_t a_idx = 0;
                    size_t b_idx = 0;
                    size_t c_idx = 0;
                    size_t idx = i;
                    for (int64 j = num_axes - 1; j >= 0; j--)
                    {
                        size_t k = idx % shape[j];
                        a_idx += (k >= a.shape[j] ? 0 : k) * a.steps[j];
                        b_idx += (k >= b.shape[j] ? 0 : k) * b.steps[j];
                        c_idx += k * c.steps[j];
                        idx /= shape[j];
                    }
                    c[c_idx] = op(a[a_idx], b[b_idx]);
                }
            }
        }

        BinaryOp::BinaryOp() : Layer("BinaryOp") { op_type = ADD; }

        void BinaryOp::Set(const std::string& key, const ParamValue& value)
        {
            if (key == "op") op_type = value;
        }

        void BinaryOp::Forward(const std::vector<Tensor>& bottoms, std::vector<Tensor>& tops, const Option& opt) const
        {
            const Tensor& a = bottoms[0];
            const Tensor& b = bottoms[1];

            Tensor& c = tops[0];

            auto a_shape = a.shape;
            auto b_shape = b.shape;
            auto a_steps = a.steps;
            auto b_steps = b.steps;
            // to insert 
            if (a_shape.size() > b_shape.size())
            {
                for (int i = 0; i < a_shape.size() - b_shape.size(); i++)
                {
                    b_steps.Insert(0, b_steps[0] * b_steps[0]);
                    b_shape.Insert(0, 1);
                }
            }
            else
            {
                for (int i = 0; i < b_shape.size() - a_shape.size(); i++)
                {
                    a_steps.Insert(0, a_steps[0] * b_steps[0]);
                    a_shape.Insert(0, 1);
                }
            }

            size_t dims = a_shape.size();
            // to chekc the shape
            for (int i = 0; i < dims; i++)
            {
                CHECK(a_shape[i] == b_shape[i] || (a_shape[i] == 1 || b_shape[i] == 1)) << "can not broadcast on " 
                    << i << " dims (" << a_shape[i] << " vs " << b_shape[i] << ")";
            }

            CHECK_EQ(a_shape.vol(), a.shape.vol());
            CHECK_EQ(b_shape.vol(), b.shape.vol());

            Tensor _a = Tensor(a_shape, Depth::D4, Packing::CHW, a.data, a_steps);
            Tensor _b = Tensor(b_shape, Depth::D4, Packing::CHW, b.data, b_steps);

            if (ADD == op_type) return Operator<BinaryAdd>(_a, _b, c, opt);
            if (SUB == op_type) return Operator<BinarySub>(_a, _b, c, opt);
            if (MUL == op_type) return Operator<BinaryMul>(_a, _b, c, opt);
            if (DIV == op_type) return Operator<BinaryDiv>(_a, _b, c, opt);
        }
	}
}