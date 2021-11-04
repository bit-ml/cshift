from graph.edges.graph_edges import Edge


class MultiDomainGraph:
    def __init__(self, config, experts, device, iter_no, silent=False):
        super(MultiDomainGraph, self).__init__()
        self.experts = experts
        self.init_nets(experts, device, silent, config, iter_no)
        print("==================")

    def init_nets(self, all_experts, device, silent, config, iter_no):
        only_edges_to_dst = config.get('GraphStructure', 'only_edges_to_dst')

        self.edges = []
        for i_idx, expert_i in enumerate(all_experts.methods):
            for expert_j in all_experts.methods:
                if expert_i != expert_j:
                    if expert_j.identifier != only_edges_to_dst:
                        continue

                    bs_test = 20
                    bs_train = 20

                    print("Add edge [%15s To: %15s]" %
                          (expert_i.identifier, expert_j.identifier),
                          end=' ')

                    new_edge = Edge(config,
                                    expert_i,
                                    expert_j,
                                    device,
                                    silent,
                                    bs_train=bs_train,
                                    bs_test=bs_test)
                    self.edges.append(new_edge)
