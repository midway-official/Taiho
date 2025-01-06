// dns.cpp
#include "DNS.h"

// 全局变量定义
int n_x0, n_y0;
double dx, dy, vx;
double velocity;
double l2_norm_x = 0.0, l2_norm_y = 0.0, l2_norm_p = 0.0;
double a, b;


// Mesh 类的构造函数
Mesh::Mesh(int n_y, int n_x)
    : u(n_y + 2, n_x + 2), u_star(n_y + 2, n_x + 2),u0(n_y + 2, n_x + 2),
      v(n_y + 2, n_x + 2), v_star(n_y + 2, n_x + 2),v0(n_y + 2, n_x + 2), 
      p(n_y + 2, n_x + 2), p_star(n_y + 2, n_x + 2), p_prime(n_y + 2, n_x + 2),
      u_face(n_y + 2, n_x + 1), v_face(n_y + 1, n_x + 2),bctype(n_y + 2, n_x + 2),zoneid(n_y + 2, n_x + 2) ,interid(n_y + 2, n_x + 2),nx(n_x), ny(n_y){}

// 初始化所有矩阵为零
void Mesh::initializeToZero() {
    u.setZero();
    u_star.setZero();
    v.setZero();
    v_star.setZero();
    p.setZero();
    p_star.setZero();
    p_prime.setZero();
    u_face.setZero();
    v_face.setZero();
    u0.setZero();   
    v0.setZero();
   
}

// 显示矩阵内容
void Mesh::displayMatrix(const MatrixXd& matrix, const string& name) const {
    cout << name << ":\n" << matrix << "\n";
}

// 显示所有矩阵
void Mesh::displayAll() const {
    displayMatrix(u, "u");
    displayMatrix(u_star, "u_star");
    displayMatrix(v, "v");
    displayMatrix(v_star, "v_star");
    displayMatrix(p, "p");
    displayMatrix(p_star, "p_star");
    displayMatrix(p_prime, "p_prime");
    displayMatrix(u_face, "u_face");
    displayMatrix(v_face, "v_face");
}

void Mesh::createInterId() {
    interid = MatrixXi::Zero(bctype.rows(), bctype.cols());
    interi.clear();
    interj.clear();
    internumber = 0;
    int count = 1;
    // 从上到下，从左到右遍历
    for(int i = 0; i < bctype.rows(); i++) {
        for(int j = 0; j < bctype.cols(); j++) {
            if(bctype(i,j) == 0) {
                interid(i,j) = count;
                interi.push_back(i);
                interj.push_back(j);
                count++;
                internumber++;
            }
        }
    }
}
void Mesh::setBlock(int x1, int y1, int x2, int y2, double bcValue, double zoneValue) {
    // 确保坐标范围合法
    x1 = std::max(0, std::min(x1, nx + 1));
    x2 = std::max(0, std::min(x2, nx + 1));
    y1 = std::max(0, std::min(y1, ny + 1));
    y2 = std::max(0, std::min(y2, ny + 1));
    
    // 确保 x1 <= x2 且 y1 <= y2
    if(x1 > x2) std::swap(x1, x2);
    if(y1 > y2) std::swap(y1, y2);
    
    // 修改指定区域的 bctype 和 zoneid
    bctype.block(y1, x1, y2-y1+1, x2-x1+1).setConstant(bcValue);
    zoneid.block(y1, x1, y2-y1+1, x2-x1+1).setConstant(zoneValue);
}
void Mesh::setZoneUV(int zoneIndex, double u, double v) {
    // 确保 zoneu 和 zonev 向量足够长
    while(zoneu.size() <= zoneIndex) {
        zoneu.push_back(0.0);
        zonev.push_back(0.0);
    }
    
    // 设置指定索引的值
    zoneu[zoneIndex] = u;
    zonev[zoneIndex] = v;
}
void Mesh::initializeBoundaryConditions() 
{
    // 遍历所有网格点，处理非内部点的速度
    for(int i = 0; i <= ny + 1; i++) {
        for(int j = 0; j <= nx + 1; j++) {
            if(bctype(i,j) > 0) {  // 非内部点
                int zone = zoneid(i,j);
                u(i,j) = zoneu[zone];
                u_star(i,j) = zoneu[zone];
                v(i,j) = zonev[zone];
                v_star(i,j) = zonev[zone];
            }
        }
    }

    // 处理 u_face
for(int i = 0; i <= ny + 1; i++) {
    for(int j = 0; j <= nx; j++) {
        // 检查面两侧的单元格
        bool left_is_internal = (bctype(i,j) == 0);
        bool right_is_internal = (bctype(i,j+1) == 0);

        if(left_is_internal && !right_is_internal) {
            // 右侧是边界，使用右侧单元格的速度
            u_face(i,j) = zoneu[zoneid(i,j+1)];
        }
        else if(!left_is_internal && right_is_internal) {
            // 左侧是边界，使用左侧单元格的速度
            u_face(i,j) = zoneu[zoneid(i,j)];
        }
        else if(!left_is_internal && !right_is_internal) {
            // 两侧都是边界，取均值
            u_face(i,j) = 0.5 * (zoneu[zoneid(i,j)] + zoneu[zoneid(i,j+1)]);
        }
    }
}

// 处理 v_face
for(int i = 0; i <= ny; i++) {
    for(int j = 0; j <= nx + 1; j++) {
        // 检查面上下的单元格
        bool top_is_internal = (bctype(i,j) == 0);
        bool bottom_is_internal = (bctype(i+1,j) == 0);

        if(top_is_internal && !bottom_is_internal) {
            // 下侧是边界，使用下侧单元格的速度
            v_face(i,j) = zonev[zoneid(i+1,j)];
        }
        else if(!top_is_internal && bottom_is_internal) {
            // 上侧是边界，使用上侧单元格的速度
            v_face(i,j) = zonev[zoneid(i,j)];
        }
        else if(!top_is_internal && !bottom_is_internal) {
            // 上下都是边界，取均值
            v_face(i,j) = 0.5 * (zonev[zoneid(i,j)] + zonev[zoneid(i+1,j)]);
        }
    }
}
}
// Equation 类的构造函数
Equation::Equation(Mesh& mesh_)
    : A_p(mesh_.ny + 2, mesh_.nx + 2),
      A_e(mesh_.ny + 2, mesh_.nx + 2),
      A_w(mesh_.ny + 2, mesh_.nx + 2),
      A_n(mesh_.ny + 2, mesh_.nx + 2),
      A_s(mesh_.ny + 2, mesh_.nx + 2),
      source(mesh_.internumber),
      A(mesh_.internumber, mesh_.internumber),
      n_x(mesh_.nx), 
      n_y(mesh_.ny),
      mesh(mesh_)
{}
// 初始化矩阵和源向量为零
void Equation::initializeToZero() {
    A_p.setZero();
    A_e.setZero();
    A_w.setZero();
    A_n.setZero();
    A_s.setZero();
    source.setZero();
    A.setZero();
}
void Equation::build_matrix() {
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;

    // 遍历所有网格点
    for(int i = 1; i <=n_y ; i++) {
        for(int j = 1; j <= n_x; j++) {
            // 只处理内部点（bctype为0的点）
            if(mesh.bctype(i,j) == 0) {
                int current_id = mesh.interid(i,j) - 1;  // 当前点在方程组中的编号
                
                // 添加中心点系数
                tripletList.emplace_back(current_id, current_id, A_p(i,j));
                
                // 检查东邻接单元
                if(mesh.bctype(i,j+1) == 0) {
                    int east_id = mesh.interid(i,j+1) - 1;
                    tripletList.emplace_back(current_id, east_id, -A_e(i,j));
                }
                
                // 检查西邻接单元
                if(mesh.bctype(i,j-1) == 0) {
                    int west_id = mesh.interid(i,j-1) - 1;
                    tripletList.emplace_back(current_id, west_id, -A_w(i,j));
                }
                
                // 检查北邻接单元
                if(mesh.bctype(i-1,j) == 0) {
                    int north_id = mesh.interid(i-1,j) - 1;
                    tripletList.emplace_back(current_id, north_id, -A_n(i,j));
                }
                
                // 检查南邻接单元
                if(mesh.bctype(i+1,j) == 0) {
                    int south_id = mesh.interid(i+1,j) - 1;
                    tripletList.emplace_back(current_id, south_id, -A_s(i,j));
                }
            }
        }
    }
    
    // 设置稀疏矩阵大小为内部点数量
    A.resize(mesh.internumber, mesh.internumber);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}
void solve(Equation& equation, double epsilon, double& l2_norm, MatrixXd& phi){
    // 创建解向量，长度为内部点数量
    VectorXd x(equation.mesh.internumber);

    // 根据 interid 构建初始解向量，遍历整个网格
    for(int i = 0; i <= equation.n_y + 1; i++) {
        for(int j = 0; j <= equation.n_x + 1; j++) {
            if(equation.mesh.bctype(i,j) == 0) {
                int n = equation.mesh.interid(i,j) - 1;
                x[n] = phi(i,j);
            }
        }
    }

    // 计算残差
    l2_norm = (equation.A * x - equation.source).norm();

    // 求解线性方程组
    BiCGSTAB<SparseMatrix<double>> solver;
    solver.compute(equation.A);
    solver.setTolerance(epsilon);
    x = solver.solve(equation.source);

    // 将结果写回网格，同样遍历整个网格
    for(int i = 0; i <= equation.n_y + 1; i++) {
        for(int j = 0; j <= equation.n_x + 1; j++) {
            if(equation.mesh.bctype(i,j) == 0) {
                int n = equation.mesh.interid(i,j) - 1;
                phi(i,j) = x[n];
            }
        }
    }
}
void face_velocity(Mesh& mesh, Equation& equ_u) {
    MatrixXd& u_face = mesh.u_face;
    MatrixXd& v_face = mesh.v_face;
    MatrixXd& bctype = mesh.bctype;
    MatrixXd& u = mesh.u;
    MatrixXd& v = mesh.v;
    MatrixXd& p = mesh.p;
    MatrixXd& A_p = equ_u.A_p;
    
    double alpha_uv = 10e-2;
    
    // 遍历 u_face (ny + 2, nx + 1)
    for(int i = 0; i <= mesh.ny + 1; i++) {
        for(int j = 0; j <= mesh.nx; j++) {
            if( bctype(i,j) == 0 && bctype(i,j+1) == 0) {
                u_face(i,j) = 0.5*(u(i,j) + u(i,j+1))
                    + 0.25*alpha_uv*(p(i,j+1) - p(i,j-1))*dy*dx/A_p(i,j)*dx 
                    + 0.25*alpha_uv*(p(i,j+2) - p(i,j))*dy*dx/A_p(i,j+1)*dx
                    - 0.5*alpha_uv*(1/A_p(i,j) + 1/A_p(i,j+1))*(p(i,j+1) - p(i,j))*dy*dx/dx;
            }
        }
    }
    
    // 遍历 v_face (ny + 1, nx + 2)
    for(int i = 0; i <= mesh.ny; i++) {
        for(int j = 0; j <= mesh.nx + 1; j++) {
            if( bctype(i,j) == 0 && bctype(i+1,j) == 0) {
                v_face(i,j) = 0.5*(v(i+1,j) + v(i,j))
                    + 0.25*alpha_uv*(p(i,j) - p(i+2,j))*dy/A_p(i+1,j)
                    + 0.25*alpha_uv*(p(i-1,j) - p(i+1,j))*dy/A_p(i,j)
                    - 0.5*alpha_uv*(1/A_p(i+1,j) + 1/A_p(i,j))*(p(i,j) - p(i+1,j))*dy;
            }
        }
    }
}
void pressure_function(Mesh &mesh, Equation &equ_p, Equation &equ_u)
{
    double alpha_uv = 10e-2;
    MatrixXd &u_face = mesh.u_face;
    MatrixXd &v_face = mesh.v_face;
    MatrixXd &bctype = mesh.bctype;
    MatrixXd &A_p = equ_u.A_p;
    MatrixXd &Ap_p = equ_p.A_p;
    MatrixXd &Ap_e = equ_p.A_e;
    MatrixXd &Ap_w = equ_p.A_w;
    MatrixXd &Ap_n = equ_p.A_n;
    MatrixXd &Ap_s = equ_p.A_s;
    VectorXd &source_p = equ_p.source;

    // 遍历网格点
    for(int i = 0; i <= equ_p.n_y+1; i++) {
        for(int j = 0; j <= equ_p.n_x+1; j++) {
            if(bctype(i,j) == 0) {  // 内部点
                int n = mesh.interid(i,j) - 1;
                double Ap_temp = 0;
                
                // 检查东面
                if(bctype(i,j+1) == 0) {
                    Ap_e(i,j) = 0.5*alpha_uv*(1/A_p(i,j) + 1/A_p(i,j+1))*(dy*dy);
                    Ap_temp += Ap_e(i,j);
                } else {
                    Ap_e(i,j) = 0;
                }

                // 检查西面
                if(bctype(i,j-1) == 0) {
                    Ap_w(i,j) = 0.5*alpha_uv*(1/A_p(i,j) + 1/A_p(i,j-1))*(dy*dy);
                    Ap_temp += Ap_w(i,j);
                } else {
                    Ap_w(i,j) = 0;
                }

                // 检查北面
                if(bctype(i-1,j) == 0) {
                    Ap_n(i,j) = 0.5*alpha_uv*(1/A_p(i,j) + 1/A_p(i-1,j))*(dx*dx);
                    Ap_temp += Ap_n(i,j);
                } else {
                    Ap_n(i,j) = 0;
                }

                // 检查南面
                if(bctype(i+1,j) == 0) {
                    Ap_s(i,j) = 0.5*alpha_uv*(1/A_p(i,j) + 1/A_p(i+1,j))*(dx*dx);
                    Ap_temp += Ap_s(i,j);
                } else {
                    Ap_s(i,j) = 0;
                }

                // 设置中心系数和源项
                Ap_p(i,j) = Ap_temp;
                source_p[n] = -(u_face(i,j) - u_face(i,j-1))*dy 
                             - (v_face(i-1,j) - v_face(i,j))*dx;
            }
        }
    }
}

void correct_pressure(Mesh &mesh, Equation &equ_u)
{
    MatrixXd &p = mesh.p;
    MatrixXd &p_star = mesh.p_star;
    MatrixXd &p_prime = mesh.p_prime;
    MatrixXd &bctype = mesh.bctype;
    int n_x = mesh.nx;
    int n_y = mesh.ny;

    // 遍历所有网格点
    for(int i = 0; i <= n_y + 1; i++) {
        for(int j = 0; j <= n_x + 1; j++) {
            if(bctype(i,j) > 0) {  // 边界点
                vector<double> neighbor_values;
                
                // 检查上邻居
                if(i > 0 && bctype(i-1,j) == 0) {
                    neighbor_values.push_back(p_prime(i-1,j));
                }
                
                // 检查下邻居
                if(i < n_y+1 && bctype(i+1,j) == 0) {
                    neighbor_values.push_back(p_prime(i+1,j));
                }
                
                // 检查左邻居
                if(j > 0 && bctype(i,j-1) == 0) {
                    neighbor_values.push_back(p_prime(i,j-1));
                }
                
                // 检查右邻居
                if(j < n_x+1 && bctype(i,j+1) == 0) {
                    neighbor_values.push_back(p_prime(i,j+1));
                }
                
                // 如果有内部单元格邻居，取平均值
                if(!neighbor_values.empty()) {
                    double avg = 0.0;
                    for(double val : neighbor_values) {
                        avg += val;
                    }
                    p_prime(i,j) = avg / neighbor_values.size();
                }
            }
        }
    }

    // 更新压力场
    double alpha_p = 0.3;  // 压力松弛因子
    p_star = p + alpha_p * p_prime;
}


void correct_velocity(Mesh &mesh, Equation &equ_u)
{
    MatrixXd &u = mesh.u;
    MatrixXd &v = mesh.v;
    MatrixXd &u_face = mesh.u_face;
    MatrixXd &v_face = mesh.v_face;
    MatrixXd &p_prime = mesh.p_prime;
    MatrixXd &u_star = mesh.u_star;
    MatrixXd &v_star = mesh.v_star;
    MatrixXd &bctype = mesh.bctype;
    MatrixXd &A_p = equ_u.A_p;
    int n_x = mesh.nx;
    int n_y = mesh.ny;
    double alpha_uv = 10e-2;

    // 修正 u 速度
    for(int i = 0; i <= n_y+1; i++) {
        for(int j = 0; j <= n_x+1; j++) {
            if(bctype(i,j) == 0) {  // 当前点是内部面
                double p_west, p_east;
                
                // 检查西面
                if(bctype(i,j-1) == 0) {
                    p_west = p_prime(i,j-1);
                } else {
                    p_west = p_prime(i,j);
                }
                
                // 检查东面
                if(bctype(i,j+1) == 0) {
                    p_east = p_prime(i,j+1);
                } else {
                    p_east = p_prime(i,j);
                }
                
                u_star(i,j) = u(i,j) + 0.5*alpha_uv*(p_west - p_east)*dy/A_p(i,j);
            }
        }
    }

    // 修正 v 速度
    for(int i = 0; i <= n_y+1; i++) {
        for(int j = 0; j <= n_x+1; j++) {
            if(bctype(i,j) == 0) {  // 当前点是内部面
                double p_north, p_south;
                
                // 检查北面
                if(bctype(i-1,j) == 0) {
                    p_north = p_prime(i-1,j);
                } else {
                    p_north = p_prime(i,j);
                }
                
                // 检查南面
                if(bctype(i+1,j) == 0) {
                    p_south = p_prime(i+1,j);
                } else {
                    p_south = p_prime(i,j);
                }
                
                v_star(i,j) = v(i,j) + 0.5*alpha_uv*(p_south - p_north)*dx/A_p(i,j);
            }
        }
    }

    // 修正 u_face
    for(int i = 0; i <= n_y+1; i++) {
        for(int j = 0; j <= n_x; j++) {
            if(bctype(i,j) == 0 && bctype(i,j+1) == 0) {  // 检查两侧是否都是内部面
                u_face(i,j) = u_face(i,j) + 
                    0.5*alpha_uv*(1/A_p(i,j) + 1/A_p(i,j+1))*(p_prime(i,j) - p_prime(i,j+1))*dy;
            }
        }
    }

    // 修正 v_face
    for(int i = 0; i <= n_y; i++) {
        for(int j = 0; j <= n_x+1; j++) {
            if(bctype(i,j) == 0 && bctype(i+1,j) == 0) {  // 检查上下是否都是内部面
                v_face(i,j) = v_face(i,j) + 
                    0.5*alpha_uv*(1/A_p(i,j) + 1/A_p(i+1,j))*(p_prime(i+1,j) - p_prime(i,j))*dx;
            }
        }
    }
}


void post_processing(Mesh &mseh,int n_x,int n_y,double a)
{   
    VectorXd x(n_x+2),y(n_y+2);
    x << 0,VectorXd::LinSpaced(n_x,dx/2.0,a-dx/2.0),a;
    y << 0,VectorXd::LinSpaced(n_y,dy/2.0,a-dy/2.0),a
    ;

    //保存计算结果
     std::ofstream outFile;
     outFile.open("u.dat");
     outFile << mseh.u_star;
     outFile.close();

     outFile.open("v.dat");
     outFile << mseh.v_star;
     outFile.close();

     outFile.open("p.dat");
     outFile << mseh.p_star;
     outFile.close();

     outFile.open("x.dat");
     outFile << x;
     outFile.close();

     outFile.open("y.dat");
     outFile << y;
     outFile.close();




}

void show_progress_bar(int current_step, int total_steps, double elapsed_time) {
    // 计算进度百分比
    double progress = static_cast<double>(current_step) / total_steps;
    
    // 设置进度条的宽度
    int bar_width = 50;
    
    // 计算进度条中"="的数量
    int pos = static_cast<int>(bar_width * progress);
    
    // 计算预计剩余时间
    double remaining_time = (elapsed_time / current_step) * (total_steps - current_step);
    
    // 打印进度条和相关信息
    std::cout << "[";
    
    // 绘制进度条
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) {
            std::cout << "=";  // 已完成的部分
        } else if (i == pos) {
            std::cout << ">";  // 当前进度的位置
        } else {
            std::cout << " ";  // 未完成的部分
        }
    }
    
    // 显示进度条，已用时间和预计剩余时间
    std::cout << "] " 
              << std::fixed << std::setprecision(2) << progress * 100 << "% "  // 显示进度百分比
              << "已用时间: " << std::fixed << std::setprecision(2) << elapsed_time << "秒 "  // 显示已用时间
              << "预计剩余时间: " << std::fixed << std::setprecision(2) << remaining_time << "秒\r";  // 显示预计剩余时间
    
    // 刷新输出，确保实时更新
    std::cout.flush();
}



void movement_function(Mesh &mesh, Equation &equ_u, Equation &equ_v,double re)
{   
    double alpha_uv=10e-2;
    int n,i,j;
    int n_x=equ_u.n_x;
    int n_y=equ_u.n_y;
    double D_e,D_w,D_n,D_s,F_e,F_n,F_s,F_w;

    double rho =1;
    D_e=dy/(dx*re);
    D_w=dy/(dx*re);
    D_n=dx/(dy*re);
    D_s=dx/(dy*re);

    // 引用网格变量
    MatrixXd &zoneid = mesh.zoneid;
    MatrixXd &bctype = mesh.bctype;
    MatrixXd &u= mesh.u;
    MatrixXd &v= mesh.v;
    MatrixXd &u_face= mesh.u_face;
    MatrixXd &v_face= mesh.v_face;
    MatrixXd &p= mesh.p;
    MatrixXd &p_star= mesh.p_star;
    MatrixXd &p_prime= mesh.p_prime;
    MatrixXd &u_star= mesh.u_star;
    MatrixXd &v_star= mesh.v_star;
    MatrixXd &A_p=equ_u.A_p;
    MatrixXd &A_e=equ_u.A_e;
    MatrixXd &A_w=equ_u.A_w;
    MatrixXd &A_n=equ_u.A_n;
    MatrixXd &A_s=equ_u.A_s;
    VectorXd &source_x=equ_u.source;
    VectorXd &source_y=equ_v.source;
    vector<double> zoneu=mesh.zoneu;
    vector<double> zonev=mesh.zonev;
    // 遍历网格
    for(i=0; i<=n_y+1; i++) {
        for(j=0; j<=n_x+1; j++) {
            if(bctype(i,j) == 0) {  // 内部面
                n = mesh.interid(i,j) - 1;
                
                // 计算面上流量
                F_e = dy*u_face(i,j);
                F_w = dy*u_face(i,j-1);
                F_n = dx*v_face(i-1,j);
                F_s = dx*v_face(i,j);

                double Ap_temp = 0;
               // 初始化源项
               double source_x_temp, source_y_temp;
    
               // 处理 x 方向源项
               if(bctype(i,j-1) == 0 && bctype(i,j+1) == 0) {
               source_x_temp = 0.5*alpha_uv*(p(i,j-1)-p(i,j+1))*dy;
               } else if(bctype(i,j-1) != 0 && bctype(i,j+1) == 0) {
               source_x_temp = 0.5*alpha_uv*(p(i,j)-p(i,j+1))*dy;
               } else if(bctype(i,j-1) == 0 && bctype(i,j+1) != 0) {
               source_x_temp = 0.5*alpha_uv*(p(i,j-1)-p(i,j))*dy;
               } else {
               source_x_temp = 0.0;
               }
              
    
               // 处理 y 方向源项
               if(bctype(i-1,j) == 0 && bctype(i+1,j) == 0) {
               source_y_temp = 0.5*alpha_uv*(p(i+1,j)-p(i-1,j))*dx;
               } else if(bctype(i-1,j) != 0 && bctype(i+1,j) == 0) {
               source_y_temp = 0.5*alpha_uv*(p(i+1,j)-p(i,j))*dx;
               } else if(bctype(i-1,j) == 0 && bctype(i+1,j) != 0) {
               source_y_temp = 0.5*alpha_uv*(p(i,j)-p(i-1,j))*dx;
               } else {
               source_y_temp = 0.0;
               }
               
               
                // 检查东面
                if(bctype(i,j+1) == 0) {
                    A_e(i,j) = D_e + max(0.0,-F_e);
                    Ap_temp += D_e + max(0.0,F_e);
                } else {
                    A_e(i,j) = 0;
                    Ap_temp += 2*D_e + max(0.0,F_e);
                    source_x_temp += alpha_uv*zoneu[zoneid(i,j+1)]*(2*D_e + max(0.0,-F_e));
                    source_y_temp += alpha_uv*zonev[zoneid(i,j+1)]*(2*D_e + max(0.0,-F_e));
                }

                // 检查西面
                if(bctype(i,j-1) == 0) {
                    A_w(i,j) = D_w + max(0.0,F_w);
                    Ap_temp += D_w + max(0.0,-F_w);
                } else {
                    A_w(i,j) = 0;
                    Ap_temp += 2*D_w + max(0.0,-F_w);
                    source_x_temp += alpha_uv*zoneu[zoneid(i,j-1)]*(2*D_w + max(0.0,F_w));
                    source_y_temp += alpha_uv*zonev[zoneid(i,j-1)]*(2*D_w + max(0.0,F_w));
                }

                // 检查北面 
                if(bctype(i-1,j) == 0) {
                    A_n(i,j) = D_n + max(0.0,-F_n);
                    Ap_temp += D_n + max(0.0,F_n);
                } else {
                    A_n(i,j) = 0;
                    Ap_temp += 2*D_n + max(0.0,F_n);
                    source_x_temp += alpha_uv*zoneu[zoneid(i-1,j)]*(2*D_n + max(0.0,-F_n));
                    source_y_temp += alpha_uv*zonev[zoneid(i-1,j)]*(2*D_n + max(0.0,-F_n));
                }

                // 检查南面
                if(bctype(i+1,j) == 0) {
                    A_s(i,j) = D_s + max(0.0,F_s);
                    Ap_temp += D_s + max(0.0,-F_s);
                } else {
                    A_s(i,j) = 0;
                    Ap_temp += 2*D_s + max(0.0,-F_s);
                    source_x_temp += alpha_uv*zoneu[zoneid(i+1,j)]*(2*D_s + max(0.0,F_s));
                    source_y_temp += alpha_uv*zonev[zoneid(i+1,j)]*(2*D_s + max(0.0,F_s));
                }

                A_p(i,j) = Ap_temp;
                source_x_temp += (1-alpha_uv)*A_p(i,j)*u_star(i,j);
                source_y_temp += (1-alpha_uv)*A_p(i,j)*v_star(i,j);
                // 设置源项
                source_x[n] = source_x_temp;
                source_y[n] = source_y_temp;
            }
        }
    }

    A_e = alpha_uv*A_e;
    A_w = alpha_uv*A_w;
    A_n = alpha_uv*A_n;
    A_s = alpha_uv*A_s;
    
    // 将系数复制到v方程
    equ_v.A_p = equ_u.A_p;
    equ_v.A_w = equ_u.A_w;
    equ_v.A_e = equ_u.A_e;
    equ_v.A_n = equ_u.A_n;
    equ_v.A_s = equ_u.A_s;
    
}