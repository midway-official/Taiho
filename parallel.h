#ifndef PARALLEL_H
#define PARALLEL_H

#include "DNS.h"
#include <mpi.h>
#include <omp.h>
// 声明全局变量（其他文件可访问）
extern double total_comm_time; // 通信时间
extern int total_comm_count,totalcount;   // 通信次数
extern double start_time, end_time;
// 发送矩阵列数据的函数声明
void sendMatrixColumn(const MatrixXd& src_matrix, int src_col, 
                     MatrixXd& dst_matrix, int dst_col, 
                     int target_rank, int tag = 0);

// 接收矩阵列数据的函数声明
void recvMatrixColumn(MatrixXd& dst_matrix, int dst_col,
                     int src_rank, int tag = 0);



//交互网格数据

void exchangeColumns(MatrixXd& matrix, int rank, int num_procs);

#endif // PARALLEL_H