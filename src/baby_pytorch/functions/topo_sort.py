def topo_sort(tensor):
    visited = set()
    sorted_list = []

    # An explicit stack of (node, expanded) pairs. A node is pushed first with
    # expanded=False so its children get pushed, then popped again with
    # expanded=True once all of them have been processed. This keeps the
    # post-order of the recursive version without its recursion depth limit.
    stack = [(tensor, False)]

    while stack:
        node, expanded = stack.pop()

        if expanded:
            sorted_list.append(node)
            continue

        if node in visited:
            continue

        visited.add(node)
        stack.append((node, True))

        # Push the children in reverse so they are popped left to right.
        for child in reversed(node.children):
            stack.append((child, False))

    return sorted_list
