#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include "mex.h"
#include "mat.h"
#include <matrix.h>
#include <opengm/functions/potts.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/inference/external/trws.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/external/trws.hxx>

typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;

//#define  PI 3.1415926535897932
//#define min(X,Y) (X<Y ? X : Y)

// this function maps a node (x, y) in the grid to a unique variable index 11

inline size_t variableIndex(const size_t x, const size_t y, const int nx) {
    return x + nx * y;
}

/**int nlhs = number of output arguments
 * mxArray *phls[] = pointer to an array which will hold the output data, each element is of type mxArray
 * int nrhs = number of input arguments
 * const mxArray *prhs[] = pointer to an array which holds the input data, each element is of type const mxarray
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
#define SEGMENTATION plhs[0]
    //#define T_OUT plhs[1]
//#define T_OUT_2 plhs[1]
    
#define UNARYPOTENTIALS prhs[0]
#define SMOOTHNESS_PENALTY prhs[1]
    
    
    
    /////////////////// OPENGM: BUILD MODEL ////////////////////////
    // variable initialization
    int n_labels = 2;
    int n_features = 5;
    int label_names[] = {-1,1};
    int n_row = 640; // rows
    int n_col = 480;
    int noPixels = n_row * n_col;
    
    // matlab data structures
    double *xValues = mxGetPr(UNARYPOTENTIALS);
    double C = mxGetScalar(SMOOTHNESS_PENALTY);
    
    int labelNumber = 1;
    int variableNumber = 0;
    
    // *m_unaryFactors[n_row * col + row]
    
    //mexPrintf("potential for variable %d label %d = %.3f \n",xValues[noPixels * labelNumber + variableNumber]);
    
    
    // construct a label space with noPixels variables, each having n_labels labels
    typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
    Space space(noPixels, n_labels);
    
    // construct a graphical model with explicit function, adding functions values
    typedef opengm::GraphicalModel<double, opengm::Adder, OPENGM_TYPELIST_2(opengm::ExplicitFunction<double> , opengm::PottsFunction<double>), Space> Model;
    Model energy(space);
    //mexPrintf("Unary terms\n");
    // unary terms
    // for each node i.e. for each variable add one function and one factor
    for(int i=0; i<noPixels; i++)
    {
        // add function
        const size_t shape[] = {n_labels};
        opengm::ExplicitFunction<double> func(shape, shape+1);
        for(int j=0; j<n_labels; j++) { // add unary term for each label
            func(j) = xValues[noPixels * j + i]; // i -> variable, j -> label
        }
        Model::FunctionIdentifier fid = energy.addFunction(func);
        
        // add factor
        int variable_indices[] = {i};
        energy.addFactor(fid, variable_indices, variable_indices+1);
    }
    
    // pairwise terms -> Potts
    opengm::PottsFunction<double> f(2, 4, 0, C); // penalty of 1 in case of unequal values
    Model::FunctionIdentifier fidPotts = energy.addFunction(f);
    
    //mexPrintf("Pairwise terms\n");
    // for each pair of nodes (x1, y1), (x2, y2) adjacent on the grid,
    // add one factor that connecting the corresponding variable indices
    for(size_t y = 0; y < n_col; ++y) {
        for(size_t x = 0; x < n_row; ++x) {
            if(x + 1 < n_row) { // (x, y) -> (x + 1, y)
                //size_t variableIndices[] = {variableIndex(x, y, n_row), variableIndex(x + 1, y, n_row)};
                size_t variableIndices[] = {variableIndex(x, y, n_col), variableIndex(x + 1, y, n_col)};
                std::sort(variableIndices, variableIndices + 2);
                energy.addFactor(fidPotts, variableIndices, variableIndices + 2);
            }
            if(y + 1 < n_col) { // (x, y) -> (x, y + 1)
                //size_t variableIndices[] = {variableIndex(x, y, n_row), variableIndex(x, y + 1, n_row)};
                size_t variableIndices[] = {variableIndex(x, y, n_col), variableIndex(x, y + 1, n_col)};
                std::sort(variableIndices, variableIndices + 2);
                energy.addFactor(fidPotts, variableIndices, variableIndices + 2);
            }
        }
    }
    
    
    
    // inference + energy
    //mexPrintf("Inference by TRWS\n");
    typedef opengm::external::TRWS<Model> TRWS;
    TRWS trws(energy);
    trws.infer();
    std::vector<size_t> labeling(noPixels);
    trws.arg(labeling);
    
    //LPBounder lpbound_max(gm);
    
    // prepare results for matlab
    SEGMENTATION = mxCreateNumericMatrix(307200, 1 , mxUINT32_CLASS, mxREAL);
    uint32_t* segm_ptr = (uint32_t *) mxGetPr(SEGMENTATION);
    
    for(int pxIdx = 0; pxIdx < noPixels; ++pxIdx) {
        segm_ptr[pxIdx] = labeling[pxIdx];
    }
    
    // return segmentation
    
// //     char* input_buf = mxArrayToString(prhs[0]);
// // 	//int paramNo = (int) mxGetScalar(prhs[1]);
// //     String^ sCertificate;
// //     sCertificate = gcnew String(input_buf);
// //
// // 	IrbAcs2^ irbfile = gcnew IrbAcs2();
// //     irbfile->IRBLOAD(sCertificate);
// //     int lastFrame = irbfile->getLastFrameNumber();
// // 	int firstFrame =  irbfile->getFirstFrameNumber();
// //     int frameCounter;
// // 	int noOfBytes = 307200 * sizeof(double);
// // 	double* frame;
// // 	int localIdx;
// // 	int noFrames = lastFrame - firstFrame + 1;
// //
// // 	B_OUT = mxCreateNumericMatrix(307200, noFrames , mxDOUBLE_CLASS, mxREAL);
// // 	//T_OUT = mxCreateNumericMatrix(noFrames,1,mxUINT32_CLASS, mxREAL);
// // 	T_OUT_2 = mxCreateNumericMatrix(noFrames,1,mxDOUBLE_CLASS, mxREAL);
// // 	double *B_Out_Ptr;
// // 	B_Out_Ptr = (double *) mxGetPr(B_OUT);
// //     /*long *T_Out_Ptr;
// // 	T_Out_Ptr = (long *) mxGetPr(T_OUT);*/
// // 	double *T_Out_Ptr_2;
// // 	T_Out_Ptr_2 = (double *) mxGetPr(T_OUT_2);
// //
// // 	//mexPrintf("%d\n",paramNo);
// //
// //     for(frameCounter = firstFrame; frameCounter < lastFrame +1; frameCounter++)
// //     {
// //         //mexPrintf("%d\n",frameCounter);
// // 		localIdx = frameCounter - firstFrame;
// // 		frame = (double*)  malloc(307200*sizeof(double));
// //         irbfile->setFrameNumber(frameCounter);
// // 		//irbfile->SavePixelsAsTextFile("test" + frameCounter + ".txt");
// //         int fr = irbfile->getFrameNumber();
// //         //long tt = irbfile->getRelativTimeStampMs2();
// // 		//double tt_orig = irbfile->getMilliTime();
// // 		//T_Out_Ptr[frameCounter - firstFrame] = tt;
// //
// // 		double* d = (double*)malloc(sizeof(double));
// //         irbfile->getParam(d,19); // timestamp since 30.12.1899
// // 		T_Out_Ptr_2[frameCounter - firstFrame] = *d;
// //         int noPx = irbfile->readPixelData(frame,0);
// //         //Console::WriteLine("-> {0}\n", irbfile->getFrameTimeStamp(1));
// // 		memcpy(&(B_Out_Ptr[localIdx * 307200]),frame,307200*sizeof(double));
// // 		//mexPrintf("%d\n",noPx);
// //     }
// //
// //
// //
// //     irbfile->IRBUNLOAD();
// // 	//==============
// // 	//declarations
// // 	//==============
    
    
    
}