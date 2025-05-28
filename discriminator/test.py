from discriminator.discriminator import SpamDiscriminator
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

discriminator = SpamDiscriminator.load("./spam_discriminator_model.pth", device)

discriminator.discriminate(
    [
        "加我微信：dafadsgasd",
        "你好，今天的天气不错！",
        "明天我们去新世界百货买点东西吧！",
        "明天我们去楼下超市买点东西吧！",
        "我",
        "我我我嚄我我我我我我我",
        "2005年3月毕业于上海交通大学计算机软件与理论专业，获工学硕士学位；",
        "有一种自己人的感觉",
    ]
)
