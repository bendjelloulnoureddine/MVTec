import matplotlib.pyplot as plt
from padim_module import PaDiM
from dataset import get_train_loader, get_test_images
from config import Config as C
import torch
import pytorch_lightning as pl

if __name__ == "__main__":
    model = PaDiM()
    trainer = pl.Trainer(accelerator="auto", max_epochs=1)
    trainer.fit(model, get_train_loader())

    print("🧪 Inference:")
    test_imgs = get_test_images()
    for img_path in test_imgs[:5]:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255
        img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(C.IMAGE_SIZE, C.IMAGE_SIZE))
        anomaly_map = model.infer_anomaly_map(img_tensor)

        plt.figure()
        plt.title("Anomaly Heatmap")
        plt.imshow(anomaly_map, cmap="hot")
        plt.axis("off")
        plt.show()

