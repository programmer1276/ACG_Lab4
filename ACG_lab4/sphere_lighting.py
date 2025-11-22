import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

# =============== БАЗОВЫЕ ВЫЧИСЛЕНИЯ ===================

def render_sphere(
    W, H, Wres, Hres,
    zO,
    z_scr,
    R, Cx, Cy, Cz,
    kd, ks, shininess,
    lights
):
    """
    Возвращает (img_uint8, I_float, I_max, I_min)
    img_uint8 – Hres x Wres (numpy, uint8)
    I_float   – Hres x Wres (numpy, float64), абсолютные яркости
    """
    O = np.array([0.0, 0.0, zO])
    C = np.array([Cx, Cy, Cz])

    # сетка пикселей на экране
    xs = np.linspace(-W/2, W/2, Wres, endpoint=False) + W/(2*Wres)
    ys = np.linspace(-H/2, H/2, Hres, endpoint=False) + H/(2*Hres)
    X, Y = np.meshgrid(xs, ys)

    Px = X.ravel()
    Py = Y.ravel()
    Pz = np.full_like(Px, z_scr)
    P_screen = np.stack([Px, Py, Pz], axis=1)

    # лучи
    dirs = P_screen - O
    dir_norm = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(dir_norm, 1e-8)

    OC = O - C
    a = np.sum(dirs * dirs, axis=1)
    b = 2.0 * np.sum(dirs * OC, axis=1)
    c = np.dot(OC, OC) - R**2

    discriminant = b**2 - 4*a*c
    hit_mask = discriminant >= 0.0

    t = np.full_like(a, np.inf)
    sqrtD = np.zeros_like(a)
    sqrtD[hit_mask] = np.sqrt(discriminant[hit_mask])

    t1 = (-b - sqrtD) / (2*a)
    t2 = (-b + sqrtD) / (2*a)

    t_candidate = np.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
    positive = (t_candidate > 0) & hit_mask
    t[positive] = t_candidate[positive]
    final_hit_mask = np.isfinite(t)

    P = O + dirs * t[:, np.newaxis]
    N = P - C
    N_norm = np.linalg.norm(N, axis=1, keepdims=True)
    N = N / np.maximum(N_norm, 1e-8)

    V = O - P
    V_norm = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.maximum(V_norm, 1e-8)

    I = np.zeros(P.shape[0], dtype=np.float64)

    for (lx, ly, lz, I0) in lights:
        Lpos = np.array([lx, ly, lz])
        L = Lpos - P
        L_norm = np.linalg.norm(L, axis=1, keepdims=True)
        L_dir = L / np.maximum(L_norm, 1e-8)

        cos_theta = np.sum(N * L_dir, axis=1)
        cos_theta = np.clip(cos_theta, 0.0, None)

        Hvec = L_dir + V
        H_norm = np.linalg.norm(Hvec, axis=1, keepdims=True)
        H_dir = Hvec / np.maximum(H_norm, 1e-8)
        cos_alpha = np.sum(N * H_dir, axis=1)
        cos_alpha = np.clip(cos_alpha, 0.0, None)

        I_diff = kd * I0 * cos_theta
        I_spec = ks * I0 * (cos_alpha ** shininess)
        I += I_diff + I_spec

    I[~final_hit_mask] = 0.0

    I_img = I.reshape(Hres, Wres)
    I_max = float(I_img.max())
    if np.any(I_img > 0):
        I_min = float(I_img[I_img > 0].min())
    else:
        I_min = 0.0

    if I_max > 0:
        I_norm = (I_img / I_max) * 255.0
    else:
        I_norm = I_img

    img_uint8 = np.clip(I_norm, 0, 255).astype(np.uint8)
    return img_uint8, I_img, I_max, I_min


# =============== GUI ===================

class SphereApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sphere Lighting (Блинн-Фонг)")

        self.image_label = ttk.Label(self)
        self.image_label.grid(row=0, column=0, rowspan=20, padx=5, pady=5)

        frm = ttk.Frame(self)
        frm.grid(row=0, column=1, sticky="nw", padx=5, pady=5)

        # helper
        def add_entry(r, text, default):
            ttk.Label(frm, text=text).grid(row=r, column=0, sticky="w")
            var = tk.StringVar(value=str(default))
            ent = ttk.Entry(frm, textvariable=var, width=12)
            ent.grid(row=r, column=1, sticky="w")
            return var

        row = 0
        ttk.Label(frm, text="Экран").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.W_var    = add_entry(row, "W (мм)",   800); row += 1
        self.H_var    = add_entry(row, "H (мм)",   600); row += 1
        self.Wres_var = add_entry(row, "Wres (пикс)", 600); row += 1
        self.Hres_var = add_entry(row, "Hres (пикс)", 450); row += 1
        self.zscr_var = add_entry(row, "z_scr (мм)", 0); row += 1

        ttk.Label(frm, text="Наблюдатель").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.zO_var = add_entry(row, "zO (мм)", -1000); row += 1

        ttk.Label(frm, text="Сфера").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.R_var  = add_entry(row, "R (мм)", 300); row += 1
        self.Cx_var = add_entry(row, "Cx (мм)", 0); row += 1
        self.Cy_var = add_entry(row, "Cy (мм)", 0); row += 1
        self.Cz_var = add_entry(row, "Cz (мм)", 800); row += 1

        ttk.Label(frm, text="Блинн-Фонг").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.kd_var    = add_entry(row, "kd", 0.9); row += 1
        self.ks_var    = add_entry(row, "ks", 0.8); row += 1
        self.shn_var   = add_entry(row, "shininess", 50); row += 1

        ttk.Label(frm, text="Источник 1").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.L1x_var = add_entry(row, "L1x", 3000); row += 1
        self.L1y_var = add_entry(row, "L1y", 2000); row += 1
        self.L1z_var = add_entry(row, "L1z", -500); row += 1
        self.I1_var  = add_entry(row, "I01", 500); row += 1

        ttk.Label(frm, text="Источник 2").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.L2x_var = add_entry(row, "L2x", -2000); row += 1
        self.L2y_var = add_entry(row, "L2y", 3000); row += 1
        self.L2z_var = add_entry(row, "L2z", -800); row += 1
        self.I2_var  = add_entry(row, "I02", 200); row += 1

        self.info_var = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.info_var, foreground="blue").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        btn = ttk.Button(frm, text="Render", command=self.on_render)
        btn.grid(row=row, column=0, columnspan=2, pady=5)

        self.current_photo = None  # чтобы картинка не удалялась сборщиком мусора

        # первый рендер
        self.on_render()

    def on_render(self):
        try:
            W    = float(self.W_var.get())
            H    = float(self.H_var.get())
            Wres = int(self.Wres_var.get())
            Hres = int(self.Hres_var.get())
            z_scr = float(self.zscr_var.get())
            zO   = float(self.zO_var.get())

            R    = float(self.R_var.get())
            Cx   = float(self.Cx_var.get())
            Cy   = float(self.Cy_var.get())
            Cz   = float(self.Cz_var.get())

            kd   = float(self.kd_var.get())
            ks   = float(self.ks_var.get())
            sh   = float(self.shn_var.get())

            L1x  = float(self.L1x_var.get())
            L1y  = float(self.L1y_var.get())
            L1z  = float(self.L1z_var.get())
            I01  = float(self.I1_var.get())

            L2x  = float(self.L2x_var.get())
            L2y  = float(self.L2y_var.get())
            L2z  = float(self.L2z_var.get())
            I02  = float(self.I2_var.get())

            lights = [
                (L1x, L1y, L1z, I01),
                (L2x, L2y, L2z, I02),
            ]

            img_uint8, I_float, I_max, I_min = render_sphere(
                W, H, Wres, Hres,
                zO,
                z_scr,
                R, Cx, Cy, Cz,
                kd, ks, sh,
                lights
            )

            # показываем картинку
            pil_img = Image.fromarray(img_uint8, mode="L")
            self.current_photo = ImageTk.PhotoImage(pil_img.resize((Wres, Hres)))
            self.image_label.configure(image=self.current_photo)

            # сохраняем файл
            pil_img.save("sphere_brightness_gui.png")

            self.info_var.set(f"Max = {I_max:.3g}, Min>0 = {I_min:.3g}")
            print("Изображение сохранено в sphere_brightness_gui.png")
            print("Max:", I_max, "Min>0:", I_min)

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


if __name__ == "__main__":
    app = SphereApp()
    app.mainloop()
