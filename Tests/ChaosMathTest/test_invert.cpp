#include "core.hpp"

namespace chaos
{
	TEST_CLASS(InvertTest)
	{
	public:
		InvertTest()
		{
			Logger::WriteMessage("Testing Invert");

			float buf36[] = {
				1, 1,  1,  1,   1,   1,
				1, 2,  3,  4,   5,   6,
				1, 3,  6, 10,  15,  21,
				1, 4, 10, 20,  35,  56,
				1, 5, 15, 35,  70, 126,
				1, 6, 21, 56, 126, 252 
			};
			A6 = Tensor(Shape(6, 6), Depth::D4, Packing::CHW, buf36).Clone();
			Ap = Tensor(Shape(3, 6), Depth::D4, Packing::CHW, buf36).Clone();

			float buf36inv[] = {
				  6, -15,   20,  -15,   6,  -1, 
				-15,  55,  -85,   69, -29,   5, 
				 20, -85,  146, -127,  56, -10, 
				-15,  69, -127,  117, -54,  10, 
				  6, -29,   56,  -54,  26,  -5, 
				 -1,   5,  -10,   10,  -5,   1
			};
			A6inv = Tensor(Shape(6, 6), Depth::D4, Packing::CHW, buf36inv).Clone();

			float bufpinv[] = {
				 1.5, -0.85714286,  0.17857143,
				 0.3,  0.05714286, -0.03571429,
				-0.4,  0.54285714, -0.14285714,
				-0.6,         0.6, -0.14285714,
				-0.3,  0.22857143, -0.03571429,
				 0.5, -0.57142857,  0.17857143
			};
			Apinv = Tensor(Shape(6, 3), Depth::D4, Packing::CHW, bufpinv).Clone();
			
			float buf9[] = {
				1,1,1,
				1,2,3,
				1,3,6
			};
			A3 = Tensor(Shape(3, 3), Depth::D4, Packing::CHW, buf9).Clone();

			float buf9inv[] = {
				 3, -3,  1,
				-3,  5, -2,
				 1, -2,  1
			};
			A3inv = Tensor(Shape(3, 3), Depth::D4, Packing::CHW, buf9inv).Clone();

			float buf4[] = {
				1, 1,
				1, 2
			};
			A2 = Tensor(Shape(2, 2), Depth::D4, Packing::CHW, buf4).Clone();

			float buf4inv[] = {
				 2, -1,
				-1,  1
			};
			A2inv = Tensor(Shape(2, 2),Depth::D4, Packing::CHW, buf4inv).Clone();

		}

		~InvertTest()
		{
			Logger::WriteMessage("Invert test finished");
		}

		TEST_METHOD(InvLU)
		{
			Tensor Ainv;
			Invert(A6, Ainv, DECOMP_LU);
			for (int i = 0; i < 36; i++)
			{
				Assert::AreEqual(A6inv[i], Ainv[i], 1E-2f);
			}
		}

		TEST_METHOD(InvCholesky)
		{
			Tensor Ainv;
			Invert(A6, Ainv, DECOMP_CHOLESKY);
			for (int i = 0; i < 36; i++)
			{
				Assert::AreEqual(A6inv[i], Ainv[i], FLT_EPSILON);
			}
		}

		TEST_METHOD(InvSVD)
		{
			Tensor Ainv;
			Invert(A6, Ainv, DECOMP_SVD);
			for (int i = 0; i < 36; i++)
			{
				Assert::AreEqual(A6inv[i], Ainv[i], 1E-2f);
			}
			Invert(Ap, Ainv, DECOMP_SVD);
			Assert::AreEqual((uint)6, Ainv.shape[0]);
			Assert::AreEqual((uint)3, Ainv.shape[1]);
			for (int i = 0; i < 18; i++) Assert::AreEqual(Apinv[i], Ainv[i], FLT_EPSILON * 10);
		}

		TEST_METHOD(InvEig)
		{
			Tensor Ainv;
			Invert(A6, Ainv, DECOMP_EIG);
			for (int i = 0; i < 36; i++)
			{
				Assert::AreEqual(A6inv[i], Ainv[i], 1E-2f);
			}
		}

		TEST_METHOD(InvDet)
		{
			Tensor Ainv;
			Invert(A3, Ainv);
			for (int i = 0; i < 9; i++) Assert::AreEqual(A3inv[i], Ainv[i], FLT_EPSILON);
			Invert(A2, Ainv);
			for (int i = 0; i < 4; i++) Assert::AreEqual(A2inv[i], Ainv[i], FLT_EPSILON);
		}

		Tensor A6; // test case 1 6x6
		Tensor A6inv;
		Tensor A3; // test case 2 3x3
		Tensor A3inv;
		Tensor A2; // test case 3 2x2
		Tensor A2inv;
		Tensor Ap; // test case 4 3x6
		Tensor Apinv;
	};
}