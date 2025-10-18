module Semantic_networks
export make_participants

using StatsBase
using LinearAlgebra

function get_word(degrees,size)
    """
    Selects a node ('word') from the existing network (1:size) with a probability
    proportional to its current degree (Preferential Attachment).
    
    # Arguments
    - `degrees`: Vector of current node degrees.
    - `size`: The current number of nodes in the network (i-1 in the main loop).
    
    # Returns
    - A randomly selected node index (Int).
    """
    return sample(1:size,Weights(degrees[1:size]))
end
function make_contact(network, utilities, word, size, m)
    """
    Selects 'm' neighboring nodes of a chosen 'word' node to form new connections.
    The selection probability for neighbors is based on their pre-defined 'utilities'.
        
    # Arguments
    - `network`: The current adjacency matrix.
    - `utilities`: Vector of inherent attractiveness scores for each node.
    - `word`: The central node (word) selected in the previous step.
    - `size`: The current number of nodes.
    - `m`: Number of connections to make.
        
    # Returns
    - A vector of 'm' unique node indices (Int) to connect to.
    """
    neighbors = findall(!iszero, view(network, 1:size, word))
    probabilities = utilities[neighbors]
    return sample(neighbors, Weights(probabilities), m, replace=false)
end
function create_network(n,m,utilities = ones(Float32, n)) 
    """
    Generates a Semantic network of 'n' words based on the Steyvers-Tenenbaum model A.
    # Arguments
    - `n`: Total number of nodes (words) in the network.
    - `m`: Number of connections each new node forms.
    - `utilities`: (Optional) Vector of Float32 representing the inherent utility of each node.
    # Returns
    - An n x n Float32 matrix representing the undirected adjacency network.
    """
    network=zeros(Float32,n,n)
    network[1:m,1:m].=1f0
    degrees = fill(Float32(m - 1), n)
    for i in m+1:n
        word=get_word(degrees,i-1)
        words = make_contact(network, utilities, word, i-1, m)
        @inbounds for w in words
            network[w, i]=1f0
            network[i, w]=1f0
            degrees[w] += 1f0
        end
        network[i,i]=1f0
        degrees[i] = Float32(m)
    end
    network[diagind(network)].=0f0
    return network
end 
function make_participants(num,n,m,change; utilities = ones(Float32, n))
    """
    Generates a 3D array of 'num' participant networks. Each network is based on a shared 
    'base_network' but randomized after a certain fixed point, introducing heterogeneity.
    # Arguments
    - `num`: Total number of participants to generate.
    - `n`: Total number of nodes (words) per network.
    - `m`: Number of connections each new node forms (used in network creation).
    - `change`: The proportion of words whose connections are *randomized* (1 - fixed_ratio).
    - `utilities`: (Optional) Node utilities passed to create_network.
    
    # Returns
    - An n x n x num Float32 array, where each slice is a participant's semantic network.
    """
    base_network = create_network(n, m,utilities)
    participants = Array{Float32, 3}(undef, n, n, num)

    fixed_nodes = round(Int, n * (1 - change))
    degrees_cache = sum(base_network, dims=2)[:]
    function build_partial_network(fixed_words::Int)
        new_net = zeros(Float32, n, n)
        new_net[1:fixed_words, 1:fixed_words] .= base_network[1:fixed_words, 1:fixed_words]
        degrees = copy(degrees_cache)
        for i in fixed_words+1:n
            word = sample(1:i-1, Weights(degrees[1:i-1]))
            neighbors = findall(!iszero, view(new_net, 1:i-1, word))
            contacts = sample(neighbors, m, replace=false)
            for w in contacts
                new_net[w, i] = 1f0
                new_net[i, w] = 1f0
                degrees[w] += 1f0
            end
            new_net[i, i] = 1f0
            degrees[i] = Float32(m)
        end
        new_net[diagind(new_net)] .= 0f0
        return new_net
    end

    for j in 1:num
        participants[:, :, j] .= build_partial_network(fixed_nodes)
    end
    return participants
end

end