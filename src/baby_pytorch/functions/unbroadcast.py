def unbroadcast(data, target_shape):
    if len(data.shape) < len(target_shape):
        raise ValueError(f"Cannot reduce shape {data.shape} to {target_shape}")

    while len(data.shape) > len(target_shape):
        data = data.sum(axis=0)

    for idx in range(len(target_shape) - 1, -1, -1):
        if data.shape[idx] == target_shape[idx]:
            continue

        if target_shape[idx] != 1:
            raise ValueError(f"Cannot reduce shape {data.shape} to {target_shape}")

        data = data.sum(axis=idx, keepdims=True)

    return data
