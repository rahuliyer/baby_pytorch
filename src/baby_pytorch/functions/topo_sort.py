def topo_sort(tensor):
    visited = set()
    sorted_list = []

    def visit(node):
        if node in visited:
            return

        visited.add(node)
        for child in node.children:
            visit(child)

        sorted_list.append(node)

    visit(tensor)

    return sorted_list
