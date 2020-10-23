#include "core.hpp"

namespace chaos
{
	TEST_CLASS(BinaryOpTest)
	{
	public:
		BinaryOpTest()
		{
			layer = dnn::LayerRegistry::CreateLayer("BinaryOp");
		}

		TEST_METHOD(AddCase1)
		{
			float abuf[] = {1,2,3,4,5,6,7,8,9};
			float bbuf[] = {4,5,6};
			Tensor A = Tensor(Shape(3,3), Depth::D4, Packing::CHW, abuf);
			Tensor B = Tensor(Shape(3,1), Depth::D4, Packing::CHW, bbuf);
			std::vector<Tensor> tops(1);
			layer->Set("op", dnn::BinOpType::ADD);
			layer->Forward({ A,B }, tops, dnn::Option());
			Tensor& C = tops[0];
			float expected[] = {5,6,7,9,10,11,13,14,15};
			for (int i = 0; i < 9; i++)
			{
				Assert::AreEqual(expected[i], C[i], FLT_EPSILON * 10);
			}
		}

		Ptr<dnn::Layer> layer;
	};
}