def topo_sort(val):
    visited = set()
    sorted_list = []

    def visit(node):
        if node in visited:
            return

        visited.add(node)
        for child in node.children:
            visit(child)

        sorted_list.append(node)

    visit(val)

    return sorted_list
