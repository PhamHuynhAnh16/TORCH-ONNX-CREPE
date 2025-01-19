import os, sys, torch, librosa

sys.path.append(os.getcwd())

from .crepe import Crepe
from .decode import viterbi
from .convert import frequency_to_bins

CENTS_PER_BIN, MAX_FMAX, PITCH_BINS, SAMPLE_RATE, WINDOW_SIZE = 20, 2006, 360, 16000, 1024  

def predict(audio, sample_rate, hop_length=None, fmin=50, fmax=MAX_FMAX, model='full', return_periodicity=False, batch_size=None, device='cpu', pad=True, providers=None, onnx=False):
    results = []

    if onnx:
        import onnxruntime as ort

        session = ort.InferenceSession(os.path.join("assets", f"crepe_{model}.onnx"), providers=providers)

        for frames in preprocess(audio, sample_rate, hop_length, batch_size, device, pad):
            result = postprocess(torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: frames.cpu().numpy()})[0].transpose(1, 0)[None]), fmin, fmax, return_periodicity)
            results.append((result[0], result[1]) if isinstance(result, tuple) else result)

        del session

        if return_periodicity:
            pitch, periodicity = zip(*results)
            return torch.cat(pitch, 1), torch.cat(periodicity, 1)

        return torch.cat(results, 1)
    else:
        with torch.no_grad():
            for frames in preprocess(audio, sample_rate, hop_length, batch_size, device, pad):
                result = postprocess(infer(frames, model, device, embed=False).reshape(audio.size(0), -1, PITCH_BINS).transpose(1, 2), fmin, fmax, return_periodicity)
                results.append((result[0].to(audio.device), result[1].to(audio.device)) if isinstance(result, tuple) else result.to(audio.device))

        if return_periodicity:
            pitch, periodicity = zip(*results)
            return torch.cat(pitch, 1), torch.cat(periodicity, 1)
        
        return torch.cat(results, 1)

def infer(frames, model='full', device='cpu', embed=False):
    if not hasattr(infer, 'model') or not hasattr(infer, 'capacity') or (hasattr(infer, 'capacity') and infer.capacity != model): load_model(device, model)
    infer.model = infer.model.to(device)

    return infer.model(frames, embed=embed)

def load_model(device, capacity='full'):
    infer.capacity = capacity
    infer.model = Crepe(capacity)
    infer.model.load_state_dict(torch.load(os.path.join("assets", f"crepe_{capacity}.pth"), map_location=device))
    infer.model = infer.model.to(torch.device(device))
    infer.model.eval()

def postprocess(probabilities, fmin=0, fmax=MAX_FMAX, return_periodicity=False):
    probabilities = probabilities.detach()

    probabilities[:, :frequency_to_bins(torch.tensor(fmin))] = -float('inf')
    probabilities[:, frequency_to_bins(torch.tensor(fmax), torch.ceil):] = -float('inf')

    bins, pitch = viterbi(probabilities)

    if not return_periodicity: return pitch
    return pitch, periodicity(probabilities, bins)

def preprocess(audio, sample_rate, hop_length=None, batch_size=None, device='cpu', pad=True):
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    if sample_rate != SAMPLE_RATE:
        audio = torch.tensor(librosa.resample(audio.detach().cpu().numpy().squeeze(0), orig_sr=sample_rate, target_sr=SAMPLE_RATE, res_type="soxr_vhq"), device=audio.device).unsqueeze(0)
        hop_length = int(hop_length * SAMPLE_RATE / sample_rate)

    if pad:
        total_frames = 1 + int(audio.size(1) // hop_length)
        audio = torch.nn.functional.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    else: total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)

    batch_size = total_frames if batch_size is None else batch_size

    for i in range(0, total_frames, batch_size):
        frames = torch.nn.functional.unfold(audio[:, None, None, max(0, i * hop_length):min(audio.size(1), (i + batch_size - 1) * hop_length + WINDOW_SIZE)], kernel_size=(1, WINDOW_SIZE), stride=(1, hop_length))
        frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE).to(device)
        frames -= frames.mean(dim=1, keepdim=True)
        frames /= torch.max(torch.tensor(1e-10, device=frames.device), frames.std(dim=1, keepdim=True))

        yield frames

def periodicity(probabilities, bins):
    probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)
    periodicity = probs_stacked.gather(1, bins.reshape(-1, 1).to(torch.int64))
    
    return periodicity.reshape(probabilities.size(0), probabilities.size(2))