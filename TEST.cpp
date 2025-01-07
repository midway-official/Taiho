#include "DNS.h"
#include <filesystem>
#include <chrono>
namespace fs = std::filesystem;

int main ()
{


    std::cout << "x方向上划分个数:";
    std::cin >> n_x0;
    
    std::cout << "y方向上划分个数:";
    std::cin >> n_y0;
    //a，b为网格边长

    std::cout << "正方形网格长度:";
    std::cin >> a;
    
    dx=a/n_x0;
    dy=a/n_y0;
    
    std::cout << "顶盖速度:";
    std::cin >> vx;
    //时间步长
    double dt;
    std::cout << "时间步长:";
    std::cin >> dt;
    //总时间步数
    int timesteps;
    std::cout << "时间步长数:";
    std::cin >> timesteps;
    
    //创建网格
    Mesh mesh(n_y0,n_x0);
    // 设置四周边界条件 bctype=1
    mesh.setBlock(0, 0, mesh.nx+1, 0, 1, 2);      // 下边界
    mesh.setBlock(0, mesh.ny+1, mesh.nx+1, mesh.ny+1, 1, 1);  // 上边界(顶盖)
    mesh.setBlock(0, 0, 0, mesh.ny+1, 1, 3);      // 左边界
    mesh.setBlock(mesh.nx+1, 0, mesh.nx+1, mesh.ny+1, 1, 1);  // 右边界

    // 设置各区域速度
    mesh.setZoneUV(0, 0.0, 0.0);  // 默认值
    mesh.setZoneUV(1, 0.0, 0.0);  // 固定壁面
    mesh.setZoneUV(2, vx, 0.0);   // 顶盖速度
    mesh.setZoneUV(3, 0, -vx);   // 顶盖速度
    // 初始化边界条件
    mesh.initializeBoundaryConditions();
    //初始化网格id
    mesh.createInterId();
    mesh.saveToFolder("lid_driven_cavity" );

}