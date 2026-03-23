import modes as ms










wavelength = 1.55       # 波长，单位 um
width = 1.0            # 波导顶宽/定义宽度，单位 um
thickness = 0.6        # 芯层厚度，单位 um
slab_thickness = 0.3   # 条形波导可设为 0；rib 波导可设成例如 0.09
angle = 80.0            # 侧壁角，90 度是矩形，小于 90 度可表示梯形侧壁

# 背景与包层设置
sub_thickness = 2.0
clad_thickness = [2.0]
sub_width = 5.0

# 折射率：这里先用常数，便于直接跑
# n_LiTaO3 = ms.materials.litao3(wavelength, axis='o')
n_LiTaO3 = ms.materials.si3n4(wavelength)
n_sio2 = ms.materials.sio2(wavelength)


# -------------------------
# 2) 调用全矢量模式求解器
# -------------------------
solver = ms.mode_solver_full(
    n_modes=4,                 # 求前 4 个模式
    width=width,
    thickness=thickness,
    slab_thickness=slab_thickness,
    angle=angle,
    wavelength=wavelength,
    x_step=0.02,
    y_step=0.02,
    sub_thickness=sub_thickness,
    clad_thickness=clad_thickness,
    sub_width=sub_width,
    n_wg=n_LiTaO3,
    n_sub=n_sio2,
    n_clads=[1.0],
    plot=False,
    overwrite=True,
)

# -------------------------
# 3) 看一下结果里有什么
# -------------------------
print("results keys:", solver.results.keys())

# 很多模式求解器都会把 neff 存在 results['neffs'] 或相近字段里
# 这里做一个兼容性读取
neff = None
for key in ["n_effs", "n_eff", "neff", "effective_index", "effective_indices"]:
    if key in solver.results:
        neff = solver.results[key]
        print(f"Found effective indices in solver.results['{key}']")
        print(neff)
        break

# if neff is None:
#     print("\n没有直接在常见字段里找到 neff。")
#     print("你可以执行下面这句查看完整结果结构：")
#     print("print(solver.results)")
# else:
#     print("\n计算得到的有效折射率：")
#     for i, val in enumerate(neff):
#         print(f"mode {i}: n_eff = {val}")
