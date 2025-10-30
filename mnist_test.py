import os
import argparse
import torch
from data_loader import load_mnist
# from models.cnn_gap import CNN
# from models.cnn_grouped_gap import CNN
# from models.cnn_pointwise_gap import CNN

# from models.cnn_depthwise_gap  import CNN
# from models.cnn_strided_depthwise_gap import CNN
# from models.cnn_strided_depthwise_dilated_gap import CNN

from models.cnn_residual_gap  import CNN

from torch.utils.data import  DataLoader

# OpenMP runtime 중복 로드 문제 우회 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
plt.close("all")

#%%
def show_sample(x_test_raw, y_pred, y_true, idx: int=0):
    """
    x_test_raw: 원본(28×28) 이미지 배열, shape=(N,28,28)
    y_pred, y_true: 1차원 리스트/배열
    """
    plt.figure()
    plt.imshow(x_test_raw[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}, True: {y_true[idx]}")
    plt.axis('off')
    plt.show()

#%%
def test(model, test_loader, device=None):
    model.eval()
    correct = 0
    total   = len(test_loader.dataset)
    all_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            if device:
                data, target = data.to(device), target.to(device)

            output = model(data)
            pred   = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.squeeze().cpu().tolist())

    acc = 100.0 * correct / total
    return acc, all_preds


#%%
def main(args):
    # DataLoader 준비
    _, _, test_ds, in_dim, x_test_raw, y_test = load_mnist( use_small=args.use_small, valDB_portion = 0.1 )
    test_loader  = DataLoader(test_ds,   batch_size=args.batch_size,   shuffle=False)

    # 모델 경로 조합
    model_path = os.path.join(args.target_model_dir, f"epoch_{args.target_epoch}.pth")
    print(f"Loading model from: {model_path}")

    # 모델 초기화 및 가중치 로드
    device = torch.device(args.device) if args.device else torch.device('cpu')
    model = CNN()
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    # 테스트 수행
    test_acc, y_pred = test(model, test_loader, device=device)
    print(f"Test Accuracy: {test_acc:.2f}%")


    show_sample(x_test_raw, y_pred, y_test, idx=args.sample_idx)


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained model on MNIST")
    parser.add_argument(
    "--target_model_dir",
    type=str,
    default="trained_output/cnn_gap_b32_lr0.01",
    help="Directory containing the trained model files"
)

    parser.add_argument("--target_epoch",      type=int,   default=21, help="Epoch number of the model to test")
    parser.add_argument("--batch_size", type=int,   default=64, help="mini-batch size for testing")
    parser.add_argument("--device",     type=str,   default=None, help="device for inference (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--sample_idx", type=int,   default=11, help="index of the test sample to visualize")
    parser.add_argument("--use_small",  type=bool,  default=False, help="use 7x7 down-sampled input")
    
    args, _ = parser.parse_known_args()

    main(args)