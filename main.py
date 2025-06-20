@app.post("/analyze-skin/")
async def analyze_skin(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_class = torch.max(probs, 0)

    # Grad-CAM logic from Part 3
    model.zero_grad()
    features = []
    grads = []

    def forward_hook(module, input, output):
        features.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0].detach())

    final_conv = model.layer4[1].conv2
    handle_f = final_conv.register_forward_hook(forward_hook)
    handle_b = final_conv.register_backward_hook(backward_hook)

    output = model(image_tensor)
    one_hot = torch.zeros((1, output.size()[-1]))
    one_hot[0][top_class.item()] = 1
    output.backward(gradient=one_hot)

    gradients = grads[0][0]
    activations = features[0][0]
    weights = torch.mean(gradients, dim=(1, 2))
    cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]

    cam = np.maximum(cam.numpy(), 0)
    cam = cam / cam.max()
    cam = np.uint8(255 * cam)
    cam = Image.fromarray(cam).resize((224, 224))
    cam = cam.convert("RGBA")
    orig = image.resize((224, 224)).convert("RGBA")
    heatmap = Image.blend(orig, cam, alpha=0.5)

    buffered = io.BytesIO()
    heatmap.save(buffered, format="PNG")
    cam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    handle_f.remove()
    handle_b.remove()

    return {
        "condition": CLASS_NAMES[top_class.item()],
        "confidence": float(top_prob.item()),
        "recommendation": "Please consult a dermatologist for confirmation.",
        "gradcam": cam_base64
    }
