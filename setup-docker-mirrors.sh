#!/bin/bash
# Docker 国内镜像源配置脚本

echo "=========================================="
echo "Docker 国内镜像源配置"
echo "=========================================="
echo ""

# 备份原配置
if [ -f /etc/docker/daemon.json ]; then
    echo "备份原配置文件..."
    sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%Y%m%d_%H%M%S)
    echo "✓ 备份完成"
    echo ""
fi

# 复制新配置
echo "应用新的镜像源配置..."
sudo cp /home/charles/codes/COAT/daemon.json.china /etc/docker/daemon.json

if [ $? -eq 0 ]; then
    echo "✓ 配置已更新"
    echo ""
    
    # 重启 Docker
    echo "重启 Docker 服务..."
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    
    if [ $? -eq 0 ]; then
        echo "✓ Docker 已重启"
        echo ""
        
        # 验证配置
        echo "验证镜像源配置："
        docker info 2>/dev/null | grep -A 10 "Registry Mirrors" || echo "等待 Docker 启动..."
        
        echo ""
        echo "=========================================="
        echo "配置完成！"
        echo "=========================================="
        echo ""
        echo "已配置的镜像源："
        echo "  1. 中科大镜像: https://docker.mirrors.ustc.edu.cn"
        echo "  2. Docker Proxy: https://dockerproxy.com"
        echo "  3. 南京大学镜像: https://docker.nju.edu.cn"
        echo "  4. DaoCloud: https://docker.m.daocloud.io"
        echo "  5. 其他镜像: https://docker.hlmirror.com"
        echo ""
        echo "现在可以尝试构建 Docker 镜像："
        echo "  docker build -t coat:latest ."
        echo ""
    else
        echo "✗ Docker 重启失败，请检查配置"
        exit 1
    fi
else
    echo "✗ 配置更新失败，请检查权限"
    exit 1
fi





