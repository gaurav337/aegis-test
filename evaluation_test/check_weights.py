import torch, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['AEGIS_MODEL_DIR'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

print('=== WEIGHT FILE INSPECTION ===')

# 1. UnivFD probe (ONLY 4KB - suspicious!)
probe = torch.load('models/univfd/probe.pth', map_location='cpu', weights_only=True)
print(f'\nUnivFD probe type: {type(probe)}')
if isinstance(probe, dict):
    print(f'  keys: {list(probe.keys())[:10]}')
    for k,v in list(probe.items())[:3]:
        if hasattr(v,'shape'):
            print(f'  {k}: shape={v.shape}, mean={v.float().mean():.6f}, std={v.float().std():.6f}')
elif hasattr(probe, 'shape'):
    print(f'  shape={probe.shape}, mean={probe.float().mean():.6f}, std={probe.float().std():.6f}')
else:
    print(f'  repr: {repr(probe)[:300]}')

# 2. FreqNet
ckpt = torch.load('models/freqnet/cnndetect_resnet50.pth', map_location='cpu', weights_only=False)
print(f'\nFreqNet type: {type(ckpt)}')
if isinstance(ckpt, dict):
    keys = list(ckpt.keys())[:8]
    print(f'  top-level keys: {keys}')
    sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
    if isinstance(sd, dict):
        print(f'  state_dict keys: {len(sd)}')
        for k,v in list(sd.items())[:3]:
            if hasattr(v,'shape'):
                print(f'  {k}: shape={v.shape}, mean={v.float().mean():.6f}, std={v.float().std():.6f}')

# 3. Xception
sd3 = torch.load('models/xception/xception_deepfake.pth', map_location='cpu', weights_only=False)
print(f'\nXception type: {type(sd3)}')
if isinstance(sd3, dict):
    keys = list(sd3.keys())[:8]
    print(f'  top-level keys: {keys}')
    sd = sd3.get('state_dict', sd3.get('model', sd3))
    if isinstance(sd, dict):
        print(f'  keys count: {len(sd)}')
        for k,v in list(sd.items())[:3]:
            if hasattr(v,'shape'):
                print(f'  {k}: shape={v.shape}, mean={v.float().mean():.6f}, std={v.float().std():.6f}')

# 4. SBI
sd4 = torch.load('models/sbi/efficientnet_b4.pth', map_location='cpu', weights_only=True)
print(f'\nSBI type: {type(sd4)}')
if isinstance(sd4, dict):
    keys = list(sd4.keys())[:8]
    print(f'  top-level keys: {keys}')
    sd = sd4.get('state_dict', sd4)
    if isinstance(sd, dict):
        print(f'  keys count: {len(sd)}')
        for k,v in list(sd.items())[:3]:
            if hasattr(v,'shape'):
                print(f'  {k}: shape={v.shape}, mean={v.float().mean():.6f}, std={v.float().std():.6f}')
