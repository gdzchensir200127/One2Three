def read_place_txt(file_path) -> tuple[list[float], list[list[float]], list[list[float]]]:
    # 打开并读取txt文件内容，指定编码为utf-8
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()

    # 初始化存储变量
    place_point = None
    path_points = []
    camera_points = []

    # 遍历每一行内容，提取所需数据
    for line in content:
        line = line.strip()  # 去除首尾空白字符
        if not line:
            continue  # 跳过空行

        # 判断是否为最佳放置点（格式匹配且包含"最佳放置点"标识）
        if "最佳放置点:" in line:
            # 提取方括号内的数值部分
            start = line.find('[')
            end = line.find(']')
            if start != -1 and end != -1:
                nums_str = line[start + 1:end].strip()
                # 将字符串转换为浮点数列表
                place_point = [float(num) for num in nums_str.split()]
                # print(f"Found place point: {place_point}")

        # 判断是否为路径信息（包含"路径"和"起点"/"终点"标识）
        elif ("路径" in line) and ("起点" in line or "终点" in line):
            start = line.find('[')
            end = line.find(']')
            if start != -1 and end != -1:
                nums_str = line[start + 1:end].strip()
                path_points.append([float(num) for num in nums_str.split()])
                # print(f"Found path point: {path_points[-1]}")

        # 判断是否为相机坐标（包含"相机"标识）
        elif "相机" in line:
            start = line.find('[')
            end = line.find(']')
            if start != -1 and end != -1:
                nums_str = line[start + 1:end].strip()
                camera_points.append([float(num) for num in nums_str.split()])
                # print(f"Found camera point: {camera_points[-1]}")

    # place_point:float list[x, y, z]
    # path_point:float list[list[x, y, z],] 4*3
    # camera_point:float list[list[x, y, z],]4*3
    return place_point, path_points, camera_points


def parse_scene_ply(file_path) -> tuple[list[list[float]], list[list[int]]]:
    # 读取Ply数据
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 解析顶点和面数量
    vertex_count = 0
    face_count = 0
    vertex_start = 0
    face_start = 0
    in_header = True

    for i, line in enumerate(lines):
        if in_header:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('element face'):
                face_count = int(line.split()[2])
            elif line.startswith('end_header'):
                in_header = False
                vertex_start = i + 1
                face_start = vertex_start + vertex_count
        else:
            break

    # 提取顶点数据
    vertices = []
    for i in range(vertex_start, vertex_start + vertex_count):
        vertex_data = list(map(float, lines[i].split()))
        vertices.append(vertex_data)

    # 提取面数据
    faces = []
    for i in range(face_start, face_start + face_count):
        face_data = list(map(int, lines[i].split()))
        faces.append(face_data)
    # vertices:float list[list[x1, y1, z1, r1, g1, b1, nx1, ny1, nz1, i1],]
    # faces:int list[list[count, idx1, idx2, idx3],]
    return vertices, faces


def parse_item_ply(file_path) -> tuple[list[list[float]], list[list[int]]]:
    # 读取Ply数据
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 解析顶点和面数量
    vertex_count = 0
    face_count = 0
    vertex_start = 0
    face_start = 0
    in_header = True

    for i, line in enumerate(lines):
        if in_header:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('element face'):
                face_count = int(line.split()[2])
            elif line.startswith('end_header'):
                in_header = False
                vertex_start = i + 1
                face_start = vertex_start + vertex_count
        else:
            break

    # 提取顶点数据
    vertices = []
    for i in range(vertex_start, vertex_start + vertex_count):
        vertex_data = list(map(float, lines[i].split()))
        # 将item_file中顶点的intensity属性乘以10
        # 如果前方有mask将item_file中顶点的intensity属性乘以8
        vertex_data[-1] *= 8
        vertices.append(vertex_data)

    # 提取面数据
    faces = []
    for i in range(face_start, face_start + face_count):
        face_data = list(map(int, lines[i].split()))
        faces.append(face_data)
    # vertices:float list[list[x1, y1, z1, r1, g1, b1, nx1, ny1, nz1, i1],]
    # faces:int list[list[count, idx1, idx2, idx3],]
    return vertices, faces


def parse_mask_ply(file_path) -> tuple[list[list[float]], list[list[int]]]:
    # 读取Ply数据
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 解析顶点和面数量
    vertex_count = 0
    face_count = 0
    vertex_start = 0
    face_start = 0
    in_header = True

    for i, line in enumerate(lines):
        if in_header:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('element face'):
                face_count = int(line.split()[2])
            elif line.startswith('end_header'):
                in_header = False
                vertex_start = i + 1
                face_start = vertex_start + vertex_count
        else:
            break

    # 提取顶点数据
    vertices = []
    for i in range(vertex_start, vertex_start + vertex_count):
        vertex_data = list(map(float, lines[i].split()))
        vertices.append(vertex_data)

    # 提取面数据
    faces = []
    for i in range(face_start, face_start + face_count):
        face_data = list(map(int, lines[i].split()))
        faces.append(face_data)
    # vertices:float list[list[x1, y1, z1, r1, g1, b1, nx1, ny1, nz1, i1],]
    # faces:int list[list[count, idx1, idx2, idx3],]
    return vertices, faces
