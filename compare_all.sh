set -e

echo "============================================"
echo "  TAAC2026 四模型对比实验"
echo "============================================"

echo ""
echo "[1/4] Training UniScaleFormer (base)..."
python train.py --config configs/base.yaml

echo ""
echo "[2/4] Training InterFormer..."
python train.py --config configs/interformer.yaml

echo ""
echo "[3/4] Training OneTrans..."
python train.py --config configs/onetrans.yaml

echo ""
echo "[4/4] Training HyFormer..."
python train.py --config configs/hyformer.yaml

echo ""
echo "============================================"
echo "  对比结果"
echo "============================================"
python scale_law.py --configs 
configs/base.yaml 
configs/interformer.yaml 
configs/onetrans.yaml 
configs/hyformer.yaml
BASH
chmod +x /mnt/data/taac2026_project/compare_all.sh
echo "compare_all.sh done"