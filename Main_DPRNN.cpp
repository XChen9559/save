// HumanDetectVC.cpp : Defines the entry point for the console application.
//

#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <ctime>
#include <vector>

using namespace std;

//#include "omp.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/c/c_api.h"
#include "models4.h"


#include "malloc.h"
#include "kiss_fftr.h"

//#include "sys/resource.h"
//#include "mcheck.h"

using namespace tflite;
clock_t start2,end2,start3,end3,start4,end4,start5,end5,start6,end6;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

//int threadnum = 2;
const int framelen_wb = 320;
const int framelen_swb = 640;
const int bufferlen_wb = framelen_wb * 2;
const int bufferlen_swb = framelen_swb * 2;

#define FreqNumRI 1536
#define WinSize 1536
#define FreqNum  FreqNumRI/2
#define HOPSIZE 960 //WinSize/2

#define HwinSize WinSize/2

#define OVERLAP (WinSize-HOPSIZE)
#define BufferSize (HOPSIZE*10 + OVERLAP)
#define WriteSize   HOPSIZE*10


#define DimFB 1024
#define HDimFB DimFB/2


#define ReadSize   HOPSIZE*10


static float win[WinSize] = {
#include "win_hop960.txt"
};

struct AIFilterBuffer{

	
	float in_L[BufferSize]; // [1, 9600]
	float in_R[BufferSize]; // [1, 9600]
	float in_state[5920]; // [1, 2, 408, 48]

	float out_L[BufferSize]; // [2, 9600]
	float out_R[BufferSize]; // [2, 9600]
	float out_state[5920]; // [1, 2, 10*46, 32]
	
}; // AIFilterBufferGlobal;

float input_L[10*1024];
float input_R[10*1024];
float input_state[5920];

float output_L[10*1024];
float output_R[10*1024];
float output_state[5920];

float temp_state[5920];
float temp_L[10*1024];
float temp_R[10*1024];


#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))


#define Nfft 1536
kiss_fftr_cfg cfg = kiss_fftr_alloc(Nfft, 0);
kiss_fftr_cfg cfgi = kiss_fftr_alloc(Nfft, 1);


   TfLiteModel* model;
   TfLiteInterpreterOptions* options;
   TfLiteInterpreter* interpreter;
   TfLiteTensor* input_tensor0;
   TfLiteTensor* input_tensor1;
   TfLiteTensor* input_tensor2;

   
   const TfLiteTensor* output_tensor0;
   const TfLiteTensor* output_tensor1;
   const TfLiteTensor* output_tensor2;
   

//int num=1;
void init()
{
    printf(" model init begin ");
//    if (num==0){
//        model = TfLiteModelCreateFromFile("/data/ylx/model1_20ms.tflite");
//        num=1;
//    }
//    else{
//        model = TfLiteModelCreate(smodel_tflite, sizeof(smodel_tflite));//TfLiteModelCreateFromFile("/data/ylx/model1_20ms_empty.tflite");
//        num=0;    
//    }

    //model = TfLiteModelCreateFromFile("/data/ylx/model1_20ms.tflite");
    model = TfLiteModelCreateFromFile("/home/samsung/users/heng.zhu/pretrained_model/model4_20ms/model4_se.tflite");
		
		// Build the interpreter
		//tflite::ops::builtin::BuiltinOpResolver resolver;
		//std::unique_ptr<tflite::Interpreter> interpreter;
	
  	//tflite::InterpreterBuilder(*(tflitehandle->model), tflitehandle->resolver)(&(tflitehandle->interpreter));
		//printf("interpreter init \n");
     
     options = TfLiteInterpreterOptionsCreate(); 
     
     interpreter = TfLiteInterpreterCreate(model, options);
     //printf("interpreter init \n");
//     printf(" interpreter begin: 0x%x \n",interpreter);
         // Create a weights cache that you can pass to XNNPACK delegate.
     
    
    
    
     TfLiteInterpreterAllocateTensors(interpreter);
//     
//     //printf(" interpreter after: 0x%x \n",interpreter);
//    
//        
//
    input_tensor0 =  TfLiteInterpreterGetInputTensor(interpreter, 0);
    input_tensor1 =  TfLiteInterpreterGetInputTensor(interpreter, 1);
    input_tensor2 =  TfLiteInterpreterGetInputTensor(interpreter, 2);

    
    output_tensor0 = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    output_tensor1 = TfLiteInterpreterGetOutputTensor(interpreter, 1);
    output_tensor2 = TfLiteInterpreterGetOutputTensor(interpreter, 2);
   
 
}



void AIFiltering_Forward_New(int * ReadData_L, int * ReadData_R, float weight_v, float weight_o, struct AIFilterBuffer* handle){
		
    int nfft = WinSize;
    kiss_fft_cpx *cx_out = new kiss_fft_cpx[nfft / 2 + 1];

	double* inFFTdata = new double[nfft] ; //
	double* outData = new double[nfft]; //
	float FFTbins_L[10][FreqNumRI];
	float FFTbins_R[10][FreqNumRI];
	float FFTbins_LtoNN[1][10][DimFB]; // This is input tensor, Model input shape is 1*12*512 , 
	float FFTbins_RtoNN[1][10][DimFB]; // This is input tensor, Model input shape is 1*12*512 , 
	
	//input_L = tflitehandle->interpreter->typed_input_tensor<float>(0); // L
   // input_R = tflitehandle->interpreter->typed_input_tensor<float>(1); // R
  //  input_state = tflitehandle->interpreter->typed_input_tensor<float>(2); // State
	start2 = clock();
  
  
  
	int i, j;
		
 	for (int i = 0; i < ReadSize; i++) //10*960
	{
		handle->in_L[i + OVERLAP] = (float)(ReadData_L[i] / 32768.0); // OVERLAP 576
		handle->in_R[i + OVERLAP] = (float)(ReadData_R[i] / 32768.0);
	}
		

	/* 1.--initial output buffer, casue the output of IFFT will be overlap adding to this buffer--*/

	for (int i = 0; i < OVERLAP; i++)
	{
		handle->out_L[i] = handle->out_L[i + WriteSize];
		handle->out_R[i] = handle->out_R[i + WriteSize];
	}
	for (int i = 0; i < WriteSize; i++)
	{	
		handle->out_L[i + OVERLAP] = 0;
		handle->out_R[i + OVERLAP] = 0;
	}		   
	
	
	
	/* 2. ----------- FFT to get the frequency bins of 12 frames---------------*/

	for(i=0; i< 10; i++)
	{

		// FFT for L channel
		for(j = 0; j< WinSize; j++)
		{
			inFFTdata[j] = handle->in_L[i* HOPSIZE + j]*win[j];
		}
		
		kiss_fftr(cfg, inFFTdata, cx_out);

		//FFTbins to keep the bins of fullband
		for (j = 0; j < FreqNum; j++)
		{
			FFTbins_L[i][j] = cx_out[j].r;
			FFTbins_L[i][j + FreqNum] = cx_out[j].i;
		}

		//FFTbins_toNN is tensor as NN input
		for (j = 0; j < HDimFB; j++)
		{
			FFTbins_LtoNN[0][i][j] = FFTbins_L[i][j];
			FFTbins_LtoNN[0][i][j + HDimFB] = FFTbins_L[i][j + FreqNum];
		}


		// FFT for R channel
		for (j = 0; j < WinSize; j++)
		{
			inFFTdata[j] = handle->in_R[i* HOPSIZE + j] * win[j];
		}

		kiss_fftr(cfg, inFFTdata, cx_out);


		for (j = 0; j < FreqNum; j++)
		{
			FFTbins_R[i][j] = cx_out[j].r;
			FFTbins_R[i][j + FreqNum] = cx_out[j].i;
		}

		for (j = 0; j < HDimFB; j++)
		{
			FFTbins_RtoNN[0][i][j] = FFTbins_R[i][j];
			FFTbins_RtoNN[0][i][j + HDimFB] = FFTbins_R[i][j + FreqNum];
		}

		
	}
 
    //printf(" finish FFT ");
	
	/* 3. ----------- NN processing---------------*/
	
//		printf(" input_L: 0x%x \n",input_L);
//		printf(" input_R: 0x%x \n",input_R);
//		printf(" output_L: 0x%x \n",output_L);
//		printf(" output_R: 0x%x \n",output_R);
//		printf(" input_state: 0x%x \n",input_state);
//		printf(" output_state: 0x%x \n",output_state);
      
     
	    for (int i = 0; i < 10; i++)
	    {
		   for (int j = 0; j < HDimFB; j++)
		   {
			   input_L[i*DimFB + 2*j ] = FFTbins_LtoNN[0][i][j ];//  /3
			   input_L[i*DimFB + 2*j +1] = FFTbins_LtoNN[0][i][j + HDimFB];//  /3

			   input_R[i*DimFB + 2*j] = FFTbins_RtoNN[0][i][j];//  /3
			   input_R[i*DimFB + 2*j + 1] = FFTbins_RtoNN[0][i][j + HDimFB];//  /3
		   }
	    }
         
//      printf("begin   NN processing ");
//      printf(" input_tensor0: 0x%x \n",input_tensor0); 
//      printf(" input_tensor1: 0x%x \n",input_tensor1);
//      printf(" interpreter: 0x%x \n",interpreter);

	end2 = clock();
	double cpu_time_used2 = ((double) (end2-start2))/CLOCKS_PER_SEC*1000;
	printf("runiime for fft module: %f ms\n",cpu_time_used2);
  
  		
      start3 = clock();
      
      TfLiteTensorCopyFromBuffer(input_tensor0, input_L, HDimFB *2*10* sizeof(float));
      TfLiteTensorCopyFromBuffer(input_tensor1, input_R, HDimFB *2*10*sizeof(float));
      TfLiteTensorCopyFromBuffer(input_tensor2, input_state, 5920* sizeof(float));
      end3 =clock();
      double cpu_time_used3 = ((double) (end3-start3))/CLOCKS_PER_SEC*1000;
      printf("runiime for memcopy1 : %f ms\n",cpu_time_used3);
      
      start4 = clock();
      TfLiteInterpreterInvoke(interpreter);
	    end4 =clock();
      double cpu_time_used4 = ((double) (end4-start4))/CLOCKS_PER_SEC*1000;
      printf("runiime for model invoke : %f ms\n",cpu_time_used4);
     //TFLITE_MINIMAL_CHECK(tflitehandle->interpreter->Invoke() == kTfLiteOk);	
	   
     start5 = clock();
	   TfLiteTensorCopyToBuffer(output_tensor0, output_L, HDimFB *2*10*sizeof(float));
     TfLiteTensorCopyToBuffer(output_tensor1, output_R, HDimFB *2*10*sizeof(float));
     TfLiteTensorCopyToBuffer(output_tensor2, output_state, 5920* sizeof(float));
     end5 =clock();
     double cpu_time_used5 = ((double) (end5-start5))/CLOCKS_PER_SEC*1000;
     printf("runiime for memcopy2 : %f ms\n",cpu_time_used5);
//     printf("after   NN processing ");

     start6 = clock();
	   for (int i=0; i<5920; i++){
            input_state[i] = output_state[i];        
     }
	 
	 
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < HDimFB; j++)
			{
				FFTbins_LtoNN[0][i][j] =  output_L[i*DimFB + 2 * j]; //12*DimFB+
				FFTbins_LtoNN[0][i][j + HDimFB] = output_L[i*DimFB + 2 * j + 1];//*3

				FFTbins_RtoNN[0][i][j] = output_R[i*DimFB + 2 * j];//*3
				FFTbins_RtoNN[0][i][j + HDimFB] = output_R[i*DimFB + 2 * j + 1];//*3
			}
		}
	 
	
	
	/* 4.----------- IFFT to get the waveform---------------*/
	
	for(i=0; i< 10; i++)
	{
		//  IFFT for L channel
		for (j = 0; j < HDimFB; j++)
		{
			FFTbins_L[i][j] = FFTbins_LtoNN[0][i][j]*weight_v +  FFTbins_L[i][j]*weight_o; //FFTbins_LtoNN[0][i][j] ;
			FFTbins_L[i][j + FreqNum] = FFTbins_LtoNN[0][i][j + HDimFB]*weight_v + FFTbins_L[i][j + FreqNum]*weight_o; //FFTbins_LtoNN[0][i][j + HDimFB];
		}


		for (j = 0; j < HDimFB; j++)
		{
			cx_out[j].r = FFTbins_L[i][j]; //*(weight_v+weight_o);
			cx_out[j].i = FFTbins_L[i][j + FreqNum];//*(weight_v+weight_o);
		}
        
        for (j = HDimFB; j < FreqNum; j++)
		{
			cx_out[j].r = FFTbins_L[i][j]*weight_o;
			cx_out[j].i = FFTbins_L[i][j + FreqNum]*weight_o;
		}
		
		kiss_fftri(cfgi, cx_out, outData);
		
		for (j = 0; j < WinSize; j++)
		{
			handle->out_L[i* HOPSIZE + j] = handle->out_L[i* HOPSIZE + j] + outData[j] * win[j];
		}


		//  IFFT for R channel
		for (j = 0; j < HDimFB; j++)
		{
			FFTbins_R[i][j] = FFTbins_RtoNN[0][i][j]*weight_v +  FFTbins_R[i][j]*weight_o; ///////
			FFTbins_R[i][j + FreqNum] = FFTbins_RtoNN[0][i][j + HDimFB]*weight_v + FFTbins_R[i][j + FreqNum]*weight_o; /////////
		}

        for (j = 0; j < HDimFB; j++)
		{
			cx_out[j].r = FFTbins_R[i][j]; //*(weight_v+weight_o);
			cx_out[j].i = FFTbins_R[i][j + FreqNum]; //*(weight_v+weight_o);
		}
		for (j = HDimFB; j < FreqNum; j++)
		{
			cx_out[j].r = FFTbins_R[i][j]*weight_o;
			cx_out[j].i = FFTbins_R[i][j + FreqNum]*weight_o;
		}

		kiss_fftri(cfgi, cx_out, outData);

		for (j = 0; j < WinSize; j++)
		{
			handle->out_R[i* HOPSIZE + j] = handle->out_R[i* HOPSIZE + j] + outData[j] * win[j];
		}
		
	}
	

	/* 5.----------   Noise + input---------- */
    



	
	/* 6. ---------- manage the input buffer------*/
	for (j = 0; j < OVERLAP ; j++)
	{
		handle->in_L[j] = handle->in_L[ReadSize  + j] ;
		handle->in_R[j] = handle->in_R[ReadSize + j];
	}
	
	
	for (int i = 0; i < WriteSize; i++)
	{
		//ReadData_R[i] = (short)(handle->out_R[i] * 32768);	
		//ReadData_L[i] = (short)(handle->out_L[i] * 32768);	
   ReadData_R[i] = (short)(min(max(handle->out_R[i] * 32768, -32768 ),32767));	
   ReadData_L[i] = (short)(min(max(handle->out_L[i] * 32768, -32768 ),32767));	
	}
	end6 =clock();
  double cpu_time_used6 = ((double) (end6-start6))/CLOCKS_PER_SEC*1000;
  printf("runiime for merger out and ifft module : %f ms\n",cpu_time_used6);
	
	
}
		



static int frameCount = 0;
int main() {

	//TFLiteHandle tflitehandle;
 
	AIFilterBuffer AIFilterBufferGlobal;
	static  FILE *f_stream , *out_stream;
	int size;
	int ReadData_L[ReadSize];
	int ReadData_R[ReadSize];
    short Data_Stereo[ReadSize*2];
    float weight_v, weight_o;
	clock_t start,end;
  double cpu_time_used;
//  struct mallinfo info1, info11, info111;
   

	// if (argc != 4) {
	// 	printf("test_tflite <tflite down or upmodel> <input pcm> <output pcm>\n");
	// 	return 1;

	// }

	

	// const char* inputpcmname = argv[1];
	// const char* outputpcmname = argv[2];	
	// const char* modelfilename = argv[3];
  //info1 = mallinfo();
  //struct rusage r_usage1=getruage(RUSAGE_SELF);
	printf("=== test_tflite Begin ===\n");  
 // printf("start  %d  \r\n",info1.uordblks);  //结果保存单位是B，可以除以1000保存为kb格式  
  //printf("start  %f %f \r\n",r_usage1.ru_maxrss/1000/1000);  //结果保存单位是B，可以除以1000保存为kb格式  
  clock_t start1,end1;
  start1 = clock();
  init();  
  end1 = clock();
  double cpu_time_used1 = ((double) (end1-start1))/CLOCKS_PER_SEC*1000;
  printf("runiime for init model: %f ms\n",cpu_time_used1);
  //info11 = mallinfo(); 
  
  //printf("model1 init %d  \r\n",info11.uordblks);

  printf(" interpreter after init: 0x%x \n",interpreter);
  printf(" main finish init\n");
  
  
// 
//			
    if ((f_stream = fopen("/home/samsung/users/heng.zhu/48k_2ch_32bit/[K_Pop]MC_mong.pcm", "rb")) == NULL)
    //if ((f_stream = fopen("/data/ylx/Vlog__1_01_mix_48k4_v18.pcm", "rb")) == NULL)
		{
			printf("Error: input bitstream file test.pcm cannot be opened\n\n");
			return -1;

		}
		
		// printf("inputfilename %s\n",inputpcmname);


   if ((out_stream = fopen("/home/samsung/users/heng.zhu/48k_2ch_32bit/[K_Pop]MC_mong_out.pcm", "wb")) == NULL)
   //if ((out_stream = fopen("/data/ylx/output.pcm", "wb")) == NULL)
		{
			printf("Error: output bitstream file test_output.pcm cannot be opened\n\n");
			return -1;
		}
   printf(" main finish open file\n");
			
   start = clock();
		while (1)
		{
			
			if (fread(Data_Stereo, sizeof(short), ReadSize*2, f_stream) <= 0)
			{
				printf("finish processing\n\n");
				break;
			}
            for (int i = 0; i < ReadSize; i++)
            {
                ReadData_L[i] = Data_Stereo[2*i];
                ReadData_R[i] = Data_Stereo[2*i+1];
            }
	
            //printf("frameCount %d\n",frameCount);
            weight_v = 1; //0.5;
            weight_o = 0;//1.0;
			//printf(" interpreter: 0x%x \n",interpreter);	
			AIFiltering_Forward_New(ReadData_L, ReadData_R, weight_v, weight_o, &AIFilterBufferGlobal);
      
            for (int i = 0; i < ReadSize; i++)
            {
                Data_Stereo[2*i] = ReadData_L[i];
                Data_Stereo[2*i+1] = ReadData_R[i];
            }
  
			fwrite(Data_Stereo, sizeof(short), ReadSize*2, out_stream);

			frameCount++;
			
//			printf("frameCount %d\n",frameCount);

			}

    printf(" main finish loop\n");
			
			fclose (f_stream);
			fclose (out_stream);
			
   TfLiteInterpreterDelete(interpreter);
   TfLiteInterpreterOptionsDelete(options);
   TfLiteModelDelete(model);

   
  
  end = clock();
  cpu_time_used = ((double) (end-start))/CLOCKS_PER_SEC*1000;
  printf("runiime: %f ms\n",cpu_time_used);
	return 0;
}
