def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)