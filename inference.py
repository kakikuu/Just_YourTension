import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# 自分の環境、条件に合わせて適切に値を代入すること
NUM_CLASSES = 7 # 分類したいクラス数　自分のタスクに合わせて正しい値を代入すること
MODEL_PATH = "ResNet152_weights.pth"
NUM_CANDIDATES = 1  # 上位2つの推論結果を表示

# 画像の前処理の設定
transform = transforms.Compose([
    transforms.Resize([224, 224]),  # 画像を224x224にリサイズ
    transforms.Grayscale(num_output_channels=1),  # グレースケールに変換
    transforms.ToTensor(),  # 画像をテンソルに変換
    transforms.Normalize(mean=[0.485], std=[0.229])  # グレースケール画像の正規化
])

# GrayScaleResNetクラスの定義
class GrayScaleResNet(nn.Module):
    def __init__(self):
        super(GrayScaleResNet, self).__init__()
        # ResNet18を使うが、事前学習は行わない
        self.resnet = models.resnet18(pretrained=False)
        # 最初の畳み込み層をグレースケール画像用に変更
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 最後の全結合層の入力特徴量を取得し、新しい全結合層を設定
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    def forward(self, x):
        return self.resnet(x)

# モデルのインスタンスを作成し、訓練済みの重みをロード
model = GrayScaleResNet()
model_path = 'ResNet152_weights.pth'  # 保存された重みのパスを指定

# 重みファイルをロードする際に、全結合層の重みとバイアスを除外
pretrained_dict = torch.load(model_path)
model_dict = model.state_dict()

# 除外する層のキーを取得（ここでは最後のfc層）
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

# 除外した状態で重みを更新
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)
model.eval()  # モデルを評価モードに設定

# クラス名のリスト
class_names = [
    "5",
    "4",
    "3",
    "2",
    "1",
    "0",
    "6",
]

# 画像を予測する関数
def predict(image_path):
    image = Image.open(image_path).convert('L')  # グレースケール画像として開く
    tensor_image = transform(image).unsqueeze(0)  # 画像をテンソルに変換し、バッチ次元を追加
    outputs = model(tensor_image)  # モデルに画像を入力し、出力を取得
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 出力をソフトマックス関数で確率に変換
    top_prob, top_classes = torch.topk(probabilities, NUM_CANDIDATES)  # 上位の確率とクラスのインデックスを取得
    top_prob_percent = [round(prob.item() * 100, 2) for prob in top_prob[0]]  # 確率をパーセンテージに変換
    # クラス名と確率を組み合わせてリストに格納
    predictions = [(class_names[class_idx], prob) for class_idx, prob in zip(top_classes[0], top_prob_percent)]
    results = []
    for class_name, prob in predictions:  # クラス名と確率を文字列に変換
        results.append(f"{class_name}")
    return results  # 予測結果のリストを返す

# # デバッグ用のmain関数
# if __name__ == "__main__":
#     image_path = "static/Uploaded_images/image.jpg"  # テストする画像のパスを指定
#     predictions = predict(image_path)  # 画像を予測する関数を呼び出す
#     for p in predictions:  # 予測結果を表示
#         print(p)
