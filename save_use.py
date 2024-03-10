def sample_architecture(self,dataset_api: dict = None, load_labeled: bool = False, op_indces:list=None) -> None:
        if load_labeled == True:
            return self.sample_random_labeled_architecture()
        def is_valid_arch(op_indices: list) -> bool:
            return not ((op_indices[0] == op_indices[1] == op_indices[2] == 1) or
                        (op_indices[2] == op_indices[4] == op_indices[5] == 1))
        while True:
            if not is_valid_arch(op_indces):
                op_indices = np.random.randint(NUM_OPS, size=(NUM_EDGES)).tolist()
                continue
            self.set_op_indices(op_indices)
            break
        self.compact = self.get_op_indices()