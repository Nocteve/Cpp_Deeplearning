vec get_full_vec(float a, int size) {
    return vec(size, a);
}
matrix get_full_matrix(float a, int h, int w) {
    return matrix(w, vec(h, a));
}