import torch
import torch.nn as nn
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# モデルの定義（Generator）
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# モデルのロード
latent_dim = 10
n_classes = 10
generator = torch.load("GANgenerator.pth", map_location=device, weights_only=False)
generator.to(device)
generator.eval()

# Streamlit UI
st.title("GAN Generator with Streamlit")
st.write("任意の数字 (0-9) に基づく手書き数字画像を生成できます。")

# ユーザー入力
label = st.number_input("生成したい数字 (0-9)", min_value=0, max_value=9, value=0, step=1)
num_images = st.slider("生成する画像数", min_value=1, max_value=10, value=5)

# 画像生成ボタン
if st.button("画像生成"):
    z = torch.randn(num_images, latent_dim, device=device)
    gen_labels = torch.tensor([label] * num_images, dtype=torch.long, device=device)

    with torch.no_grad():
        gen_imgs = generator(z, gen_labels)

    gen_imgs = (gen_imgs + 1) / 2  # [-1,1] → [0,1] にスケール

    # 画像を表示
    fig, axes = plt.subplots(1, num_images, figsize=(2 * num_images, 2))
    if num_images == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(gen_imgs[i].cpu().squeeze(), cmap="gray")
        ax.axis("off")
    st.pyplot(fig)
