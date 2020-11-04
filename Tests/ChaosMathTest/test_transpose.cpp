#include "core.hpp"

namespace chaos
{
	TEST_CLASS(TransposeTest)
	{
	public:
		TransposeTest() {}
		
		TEST_METHOD(TransposeInplace)
		{
			float a[] = { 1,2,3,4,5,6,7,8,9 };
			Tensor A = Tensor(Shape(3,3), Depth::D4, Packing::CHW, a);

			Transpose(A, A);

			float at[] = { 1,4,7,2,5,8,3,6,9 };
			for (int i = 0; i < 9; i++)
			{
				Assert::AreEqual(at[i], A[i], FLT_EPSILON);
			}

			float b[] = { 1,2,3,0,4,5,6,0,7,8,9,0 };
			Tensor B = Tensor(Shape(3, 3), Depth::D4, Packing::CHW, b, { 4,1 });
			Transpose(B, B);
			float bt[] = { 1,4,7,0,2,5,8,0,3,6,9,0 };
			for (int r = 0; r < 3; r++)
			{
				for (int c = 0; c < 3; c++)
				{
					int idx = r * 4 + c;
					Assert::AreEqual(bt[idx], B[idx], FLT_EPSILON);
				}
			}
		}

		TEST_METHOD(Transpose3x4)
		{
			float buf[] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
			Tensor A = Tensor(Shape(3,4), Depth::D4, Packing::CHW, buf);

			Tensor B;
			Transpose(A, B);
			float b[] = { 1,5,9,2,6,10,3,7,11,4,8,12 };
			for (int i = 0; i < 12; i++)
			{
				Assert::AreEqual(b[i], B[i], FLT_EPSILON);
			}

			AutoBuffer<float> cc;
			Tensor C = Tensor(Shape(4, 3), Depth::D4, Packing::CHW, cc.data(), { 4,1 });
			Transpose(A, C);
			for (int r = 0; r < 4; r++)
			{
				for (int c = 0; c < 3; c++)
				{
					int idx1 = r * 3 + c;
					int idx2 = r * 4 + c;
					Assert::AreEqual(b[idx1], cc[idx2], FLT_EPSILON);
				}
			}
		}

		TEST_METHOD(Transpose4x3)
		{
			float buf[] = {1,2,3,0,4,5,6,0,7,8,9,0,10,11,12,0};
			Tensor A = Tensor(Shape(4, 3), Depth::D4, Packing::CHW, buf, { 4,1 });

			Tensor B;
			Transpose(A, B);
			float b[] = { 1,4,7,10,2,5,8,11,3,6,9,12 };
			for (int i = 0; i < 12; i++)
			{
				Assert::AreEqual(b[i], B[i], FLT_EPSILON);
			}

			AutoBuffer<float> cc;
			Tensor C = Tensor(Shape(3, 4), Depth::D4, Packing::CHW, cc.data(), { 16,1 });
			Transpose(A, C);
			for (int r = 0; r < 3; r++)
			{
				for (int c = 0; c < 4; c++)
				{
					int idx1 = r * 4 + c;
					int idx2 = r * 16 + c;
					Assert::AreEqual(b[idx1], cc[idx2], FLT_EPSILON);
				}
			}
		}
	};
}