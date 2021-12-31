<h1 align="center">
    <b>CS231 - Introduction to Computer Vision</b>
</h1>

Giảng viên hướng dẫn: ***TS. Nguyễn Vinh Tiệp*** 


# [SEAM-CARVING FOR OBJECT REMOVAL](https://github.com/quangcrazymen/CS231_FinalProject)


![](https://i.imgur.com/yMvrK7n.gif)

## Thành viên nhóm:
**Nguyễn Đỗ Quang** - MSSV: **20520720**
**Âu Thiên Phước** - MSSV: **19522050**

[Source code repo](https://github.com/quangcrazymen/CS231_FinalProject) 
## Table of content:

[toc]



## I. Giới thiệu:

**Seam-carving** là kĩ thuật thay đổi kích thước hình ảnh, thuật toán sẽ đi tìm những đường seam (như những đường chỉ) cắt xuyên qua ảnh, và dần dần loại bỏ những đường seam này, cho đến khi tấm ảnh về kích thước mà ta mong muốn. Seam-carving tuy thay đổi kích thước của ảnh nhưng sẽ ***bảo toàn tỉ lệ của 1 số vật thể quan trọng*** (có mức năng lượng cao) trong ảnh.
    
    Input là: 1 tấm ảnh
    Output: 1 tấm ảnh đã được resize về kích thước mong muốn



![](https://i.imgur.com/99fr3Rr.png)

*Đường seam* là những đường màu đỏ trong hình

<p id="intro">

</p>

## II.Ứng dụng:
### 1. Loại bỏ đường seam (Seam_removal)
#### 1.1. Tính toán ảnh năng lượng:

Để biết thêm chi tiết tham khảo 2 bài báo [seam carving](https://faculty.idc.ac.il/arik/SCWeb/imret/index.html) và [Improved seam carving](https://faculty.idc.ac.il/arik/SCWeb/vidret/index.html).

Trước khi thức hiện seam-carving ta cần biết được mức độ quan trọng của các vùng trong ảnh. Ta sẽ thực hiện tính toán ***ảnh năng lượng***, những điểm ảnh ***ít có sự chênh lệch về màu sắc*** sẽ là những điểm ảnh được coi là ít quan trọng 
=> Được gán cho giá trị **năng lượng thấp**

![](https://i.imgur.com/goSKft6.jpg)

**Hai cách tính ảnh năng lượng:**

##### 1.1.1. Backward-Energy:

**Công thức**(Gradient-magnitude) **:** 
$$\ e_i = | \frac{\partial}{\partial x}I |+| \frac{\partial}{\partial y}I |$$
Để tính ảnh năng lượng theo phương pháp backward-energy ta để áp **Sobel filter** lên tấm ảnh  
$$ p^,_u=\begin{bmatrix}
1 & 2 & 1\\
0 & 0 & 0\\
-1 & -2 &-1
\end{bmatrix}*G , p^,_v=\begin{bmatrix}
1 & 0 & -1\\
2 & 0 & -2\\
1 & 0 & -1
\end{bmatrix}*G$$
Thực hiện phép convolution giữa tấm ảnh với lần lượt 2 kernel trên. Sau đó tổng giá trị tuyệt đối của $\ p^,_v$ và $\ p^,_u$ (đúng như công thức gradient-magnitude).

```python
def calculate_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map
```
##### 1.1.2. Forward-Energy:
Cả 2 bài báo đều đề cập tới việc những điểm ảnh có **năng lượng càng cao** thì càng ít khả năng điểm ảnh đó bị đường seam cắt qua.
Trong đề tài lần này em sẽ sử dụng cách tính ảnh năng lượng cải tiến **(forward-energy)**, được đề cập đến trong bài báo thứ 2 để tính toán ảnh năng lượng.

![](https://i.imgur.com/2F7byjd.png)

Trong bài báo thứ 2 này thay vì tạo ra ảnh năng lượng bằng cách sử dụng Sobel-filter như trên, trước khi tìm đường seam , theo phương pháp *(forward-energy)*, ở mỗi bước ta sẽ tìm luôn những đường seam có mức năng lượng tối thiểu. Những đường seam này **không nhất thiết là có năng lượng bé nhất**, mà nó sẽ là những đường seam sau khi **loại bỏ sẻ ít ảnh hưởng đến chất lượng của bức ảnh**. 
```python=51
def forward_energy(im):
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
    
    #toggle to visualize gradient magnitude img
    vis = visualizeGradient(energy)    
        
    return energy
 
```
Vector hóa tính toán (tăng tốc độ) bằng cách sử dụng np.roll

![](https://i.imgur.com/Ttt2je8.png)
Bên trái sử dụng cách tính ảnh năng lượng cũ (backward-energy), bên phải sử dụng cách tính tối ưu hơn (forward-energy)

![](https://i.imgur.com/tn6prPZ.png)

Do ảnh mức năng lượng ở vùng ảnh có cái ghế khi dùng backward-energy cao, nên các đường seam không chạy qua được, có thể khiến cho bức ảnh có những nét không tự nhiên

![](https://i.imgur.com/fcDUJul.jpg)

Bức ảnh bên trái sử dụng backward energy, xuất hiện vùng tối lạ bao quanh cái ghế, do đường seam hầu hết là cắt bên ngoài vùng ảnh của cái ghế. Bên phải (forward-energy) bức ảnh sau khi được cắt, nhìn tự nhiên hơn.

![](https://i.imgur.com/mYIRNg7.png)

Nhờ vào việc đường seam cắt ngang qua vật thể, ta có được bức ảnh có **tỉ lệ cân đối hơn giữa các vật thể**. Như bức ảnh *Starry Night* có mức năng lượng cao ở mọi nơi trên tấm ảnh, đường seam tập trung cắt chủ yếu vào bụi cây, mà ở đó mức năng lượng thấp nhất, khiến cho bụi cây, và 1 số vật thể biến dạng (**ảnh bên trái** sử dụng backward-energy). Còn **ảnh bên phải** (forward-energy), có tỉ lệ sau khi cắt giữa các vật thể được bảo toàn, cũng như ***ít bị biến dạng hơn***.


#### 1.2: Tìm đường seam có giá trị nhỏ nhất từ đỉnh đến cuối tấm ảnh (minimum-seam)


Việc tìm đường seam để loại bỏ thực chất là 1 bài toán ***quy hoạch động (dynamic-programming)***. Bước đầu tiên là đi từ đầu đến cuối tấm ảnh để tính toán năng lượng tích lũy M, để chọn ra đường seam có giá trị tích lũy bé nhất (giống với bài toán tìm đường đi ngắn nhất)

**Công thức:**
$$\ M(i,j) = e(i,j) + min(M(i-1,j-1),M(i-1,j),M(i-1,j-1))$$
```python=161
    h, w = im.shape[:2]
    energyfn = forward_energy
    M = energyfn(im)

    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy
```
#### 1.3: Dò đường và xóa đường seam (seam-removal)

Sau khi đã lấp đầy M, giá trị nhỏ nhất của hàng cuối cùng sẽ cho ta biết **đường seam có tổng giá trị năng lượng là nhỏ nhất**. Bây giờ ta có thể **backtrack lại đường đi của đường seam** này, và xóa đường seam.

```python=189
seam_index = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_index.reverse()
    return np.array(seam_idx), boolmask
```
Lặp lại bước **1.2 (Tìm đường seam)** và xóa đường seam đến khi tấm ảnh có kích thước mà mình mong muốn.

### 2. Mở rộng bức ảnh (Seam_insertion)
**Mở rộng bức ảnh** có thể coi là *ngược lại* so với thu nhỏ bức ảnh, khi ta chèn thêm đường seam vào bức ảnh. Đầu tiên ta thực hiện seam-carving cho bản sao của tấm ảnh và lưu lại các tọa độ điểm ảnh. Sau đó chèn những đường seam mới vào tấm ảnh gốc với tọa độ đã ghi lại ở trên. Các điểm ảnh mới có **giá trị bằng trung bình cộng của những điểm ảnh lân cận ở bên trái và bên phải**.

```python=95
def add_seam(im, seam_idx):
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        column = seam_idx[row]
        for ch in range(3):
            if column == 0:
                p = np.average(im[row, column: column + 2, ch])
                output[row, column, ch] = im[row, column, ch]
                output[row, column + 1, ch] = p
                output[row, column + 1:, ch] = im[row, column:, ch]
            else:
                p = np.average(im[row, column - 1: column + 1, ch])
                output[row, : column, ch] = im[row, : column, ch]
                output[row, column, ch] = p
                output[row, column + 1:, ch] = im[row, column:, ch]

    return output

```
### 3. Loại bỏ vật thể (Object_removal)
#### 3.1: Loại bỏ vật thể bằng seam-carving:
Khi tạo ra ảnh năng lượng, phần ảnh cần được xóa *(ảnh mask)* sẽ được đánh dấu với ***giá trị âm lớn***, để đảm bảo chắc chắn đường seam sẽ cắt qua các điểm ảnh ở phần mask.
```python=1
if remove_mask is not None:
    M[np.where(remove_mask > MASK_THRESHOLD)] = -ENERGY_MASK_CONST * 100
```
 Seam-carving được thực hiện liên tục đến khi phần cần xóa được loại bỏ hoàn toàn. Seam-carving cũng sẽ được thực hiện trên tấm ảnh năng lượng để có thể liên tục tạo ra bức ảnh năng lượng chính xác trong suốt quá trình xóa vật thể.
 
**Video demo:**

{%youtube A2D982hEi20 %}
#### 3.2: Mở rộng bức ảnh
Sau khi xóa xong vật thể, ta thực hiện **mở rộng bức ảnh (Seam-insertion)** về kích thước ban đầu.
## III.Kết Luận, đánh giá:
### 1. Seam-carving hoạt động tốt:

Loại bỏ vật thể sử dụng seam-carving hoạt động tốt khi có khoảng năng lượng thấp ngăn cách các phần mà mình cần xóa.

![](https://i.imgur.com/sr81NwF.png)
![](https://i.imgur.com/vTlCYn1.png)

Mặc dù tốt nhưng thuật toán vẫn có chút hạn chế, nếu như để ý kĩ thì thuật toán sẽ không phục dựng lại được những phần ảnh bị vật thể che khuất.

### 2. Seam-carving hoạt động kém hiệu quả:

Những trường hợp seam-carving xảy ra lỗi là trường hợp tấm ảnh có vật thể chiếm chọn khung hình

![](https://i.imgur.com/i15ncN1.png)
Không những đoàn tàu bị biến mất, mà tấm ảnh bị phá hủy luôn

![](https://i.imgur.com/MOkQfg1.png)
Ở bức ảnh 1(resize) và bức ảnh 2(remove object) cả 2 đều bị biến dạng.

### 3. Hướng giải quyết:
#### 3.1 Đối với những trường hợp giống như của đoàn tàu:

Thay vì cắt theo đường seam dọc ta sẽ cắt đường seam theo chiều ngang. Muốn cắt đường seam theo chiều ngang, thì chỉ cần viết thêm 1 hàm **xoay ảnh $90^o$** (cơ bản thì đường seam sẽ vẫn được cắt qua tấm ảnh theo chiều dọc), sau khi cắt xong thì xoay ảnh lại vị trí ban đầu

```python=44
def rotate_image(image, counter_clockwise): 
    k = 1 if counter_clockwise else 3
    return np.rot90(image, k)
```

**Video demo:**
{%youtube ln_H4lkL2iw%}

**Kết quả:**
![](https://i.imgur.com/UboDnpW.png)
![](https://i.imgur.com/EhkDiBE.png)

**Hạn chế:**
Mặc dù bức ảnh không bị phá hủy, nhưng kết quả ảnh cho ra cũng chưa được tự nhiên. 

#### 3.2. Đối với trường hợp có nhiều người đứng sát nhau trong bức ảnh.
![](https://i.imgur.com/EDkpdm6.png)




Ta cũng có thể tận dụng kĩ thuật sử dụng ảnh mask để bảo vệ những phần quan trọng trong tấm ảnh, bằng cách gán **giá trị năng lượng cao** vào phần ảnh cần bảo vệ
```python=165
if mask is not None:
        M[np.where(mask > MASK_THRESHOLD)] = 100000.0
```
![](https://i.imgur.com/aFUlE0e.png)
Ví dụ như bảo vệ diễn viên ở giữa màn hình

![](https://i.imgur.com/Tf6CPAV.png)
Cắt người ở giữa và bảo vệ những người xung quanh

## IV. Những cải thiện trong tương lai:
Sau khi hoàn thành xong đồ án, nhóm em nhận thấy thuật toán seam-carving là 1 thuật toán đơn giản, hiệu quả để thu nhỏ, phóng to, xóa vật thể trong những tấm ảnh có background lớn. Một số điểm cần cải tiến:
+ Cải thiện tốc độ chạy cho thuật toán
+ Tìm hiểu cách tự động hóa quá trình xóa vật thể tùy theo mỗi trường hợp mà sử dụng đường seam dọc hoặc ngang
+ Tìm hiểu những kỹ thuật và thuật toán khác để có thể khôi phục lại phần ảnh bị nhiễu sau khi xóa vật thể, do bị vật thể che khuất.

## V. Tư liệu tham khảo
*Seam Carving for Content-Aware Image Resizing(2007):*
https://faculty.idc.ac.il/arik/SCWeb/imret/index.html
*Improved Seam Carving for Video Retargeting(2008)*
https://faculty.idc.ac.il/arik/SCWeb/vidret/index.html
*Implementation of improved Seam Carving Using Backward Energy (old method):*
https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
*Implementation of improved Seam Carving Using Forward Energy:*
https://github.com/axu2/improved-seam-carving
*Implement Seam-carving to enlarge image, and remove object:*
https://github.com/vivianhylee/seam-carving.

