# VPHT Algorithm Visualization

A tool for visualizing and brute-forcing VPHT algorithm results.

This project is built using [egui](https://github.com/emilk/egui) and [eframe](https://github.com/emilk/egui/tree/master/crates/eframe).
It is based on the [eframe template](https://github.com/emilk/eframe_template/).

Live demo: https://kalokak.github.io/vpht-algo/

# Running the Tool

## Getting Started

### Prerequisites

- Rust toolchain (`rustup`)
- [Trunk](https://trunkrs.dev/) (for web build): `cargo install --locked trunk`
- Web target: `rustup target add wasm32-unknown-unknown`

### Running Locally

**Native:**
```sh
cargo run --release
```

**Web:**
```sh
trunk serve
```
Open `http://127.0.0.1:8080/index.html#dev` in a browser.

> Note: The `#dev` suffix bypasses service worker caching, ensuring the latest build is loaded.

## Acknowledgments

- [eframe_template](https://github.com/emilk/eframe_template/) for the initial template.
- [egui](https://github.com/emilk/egui) for the GUI framework.

## License 

Licensed under either of Apache-2.0 or MIT at your option.

# Implementation Detials

To complement the theoretical exploration of vertical graphs and
alternating cycles, we developed a computational tool with the purpose
of more easily investigating the relationship between topological
features of certain graphs and their VPHT. This approach was motivated
by the hypothesis that an exhaustive search on small graphs might reveal
structural patterns that obstruct unique reconstructability, as well the
question whether all non reconstructible pairs exhibit the previously
detailed cycle pattern.

#### Restriction to Non-Multiset Graphs

As brute force search on an infinite number of edges between two
vertices is impossible in finite time using any computational method
known to us, as well as the fact that graphs as simplicial complexes
restrict themselves to only allowing at most one edge between any two
vertices, call these **ordinary-graphs** here, we do the same.

For all of this section and the results contained within, we restrict
ourselves to colliding pairs of graphs, where each individual graph is
an ordinary-graph.

#### Computational Limits

Aside from the restriction to single-edge graphs stated above, we would
like to highlight some limitations of our computations. For one, any
brute force search of the kind we perform very quickly runs into a major
scaling problem. The number of possible graphs that need to be
considered on any given $n$ vertices is
$2^{\left(\frac{n(n-1)}{2}\right)}$. For us, computations on six
vertices are near instant, on seven it takes a few seconds, for eight we
could be looking at half an hour and anything beyond that is out of
reach. These scaling laws apply no matter how powerful the hardware.

Our particular implementation has a hard limit of allowing at most 32
vertices in search algorithms, however this is practically irrelevant as
the number of graphs to search there is on the order of $10^{149}$.

## Application Feature Overview 

The application allows users to draw a graph and visualize it's
persistence diagrams for 0-th and 1-st dimensional homology in a given
selected **sweep direction**. The application then allows for two basics
types of brute force search. For these brute force search methods, it is
important to note that they compute the VPHT only in the selected sweep
direction $v$ and it's reverse for a given graph $G$; this approach is
equivalent to considering the VPHT of the graph where all vertices of
$G$ are projected on the line defined by the $\mathrm{span}(v)$. This is
done because of reconstructability of general position graphs, as
outlined in the accompanying paper. For this reason, when speaking of
\"computing VPHT\" in the following, we mean computing up and down
direction diagrams. The offered methods of brute force search are:

1.  Find all possible graphs on the same set of vertices as the drawn
    graph $G$, compute their VPHT, and record all graphs $G'$ with the
    same VPHT as $G$. They are depicted as **colliding pairs** with $G$.
    This feature is called **Compute Colliding Graphs**.

2.  Find the set of all possible graphs $\Omega$ on the same set of
    vertices as the drawn graph, which include the (possibly empty) set
    of edges of the drawn graph. Within $\Omega$ search for sets of
    graphs which all share the same VPHT, so-called **collision
    sets**[^1]. The application then allows users to filter and sort
    these collision sets by number of connected components in the
    graphs[^2], number of cycles[^2], number of off diagonal points in the
    diagram, longest cycle found within a set[^3] and if the set
    contains a colliding pair which does not partition into cycles. This
    feature is called **Compute Colliding Sets**.

For both of these brute force search methods, we implemented a cycle
search available through the **Show Cycles** button. This will attempt
to partition the graphs in a collision set pairwise into cycles, if no
partition is found, non exists. Finally, there are two main
configuration options for the brute force search:

1.  **Ignore Dangling Vertices:** Determines if graphs with isolated
    vertices should be ignored for the search. Note that this may
    include the original graph for \"Compute Colliding Graphs\" in which
    case no results will be returned!

2.  **Exclude Common Edges from Cycle Search:** Determines whether to
    completely ignore edges common to both graphs[^4] for the cycle
    search in colliding pairs.

The tool features various tooltips accessible by **hovering over the
various features,** and a more in-depth explanation of the search and
sort functionality through a \"help\" button. In the coming sections, we
will detail some of the inner workings of the tool.

## Core Algorithms

Here, we briefly describe the central algorithms used for the
computational tool, which are relevant to the numerical results we
obtained. For performance reasons, the actual implementation is highly
parallel and optimizes certain parts of the procedures, but remains
computationally equivalent to what we outline here.

### Computing the VPHT 

To obtain the VPHT, we follow much of the same steps computationally as
we do mathematically in the paper. For a given sweep direction (and its reverse) we start by computing a filtration. This is done using a standard sweepline algorithm approach:

```r
Algorithm: Sweep Line Filtration
Require: Graph G=(V,E), Sweep Direction d
Ensure: Vertices V_ord sorted by height, Edges E_ord sorted by birth time.

Function BuildFiltration(G, d):
    V_ord <- List of (v, height(v, d)) for all v in V
    Sort V_ord ascending by height
    E_ord <- {}
    
    # Iterate from highest to lowest
    for i <- |V_ord| - 1 down to 1:
        (u, h_u) <- V_ord[i]
        
        # Check against all strictly lower vertices
        for j <- 0 to i - 1:
            (v, _) <- V_ord[j]
            if edge (u,v) exists in G:
                e_new <- (i, j)   # Edges store indices to V_ord
                Append (e_new, h_u) to E_ord   # Birth time is height of upper vertex
            end if
        end for
    end for

    Reverse E_ord   # Resulting order: increasing birth time
    Return (V_ord, E_ord)
End Function
```

We then use the matrix reduction algorithm to find creator destructor pairs. Out of this, we compute persistence diagrams for 0-th and 1-st homology. The persistence diagrams are sets of points with multiplicity, where the multiplicity denotes how many
creator destructor pairs share a given birth-death time. To reduce the
complexity of comparing persistence diagrams further along in the
process, we store the points sorted lexicographically first by birth
then by death time. As we assume collinear vertices (in practice graphs
projected onto the sweep direction), the diagrams for the sweep
direction and its reverse constitute the whole VPHT.

### Computing Colliding Graphs

Given that we can compute the VPHT for the subset of graphs we treat in
our work checking for colliding pairs on the set of vertices given by
the graph is a simple task. We enumerate all possible edge
combinations[^5] and compare the VPHT of the resulting graph to the VPHT
of the drawn graph. We collect all graphs which form a colliding pair
with the drawn graph, including the graph itself, and display the result
to the user. The user has the option to search for cycles for colliding
pairs in the result.

### Computing Collision Sets

Finding all possible graphs $\Omega$ which include the drawn graph's
edges remains computationally simple. However, the partition of those
graphs into colliding sets is more involved. Ultimately we are seeking
to obtain the equivalence classes of the relation \"identical VPHT\" in
$\Omega$ and any algorithm achieving this will lead to an equivalent
implementation. Nonetheless, we briefly present our approach here.

Our approach is based on the idea that one can hash data and use a
truncation of the hash to index into an array, to obtain
$\mathcal{O}(\lvert \Omega \rvert)$ quotienting, at the cost of memory.
We derive a hash for the VPHT based on the FNV1A hash algorithm. We construct an array of
\"lists-of-collision-sets\" with a power of $2$ size which is
$\mathcal{O}(\lvert \Omega \rvert)$ and for each graph index into it
using the appropriate number of bits from the hash. The resulting
list-of-collision-sets is then searched for a set which matches the
graph's VPHT, if non is found a new one is added to the list. The graph
is inserted into the appropriate set. At the end, we collect all
non-singleton collision sets.

### Searching for Alternating Cycles and Finding Minimal Partitions 

We have mentioned cycle search numerous times throughout the previous
sections. In effect, we are interested in determining if a pair of
graphs sharing a VPHT are a colliding pair (that is, their disjoint
union graph partitions into alternating cycles with the graphs being of
type $\mathcal{G}_1,\mathcal{G}_2$). Here we outline an algorithmic
approach to this.

Given two graphs $G_1, G_2$, we construct their disjoint union graph and
try to split it into alternating cycles. We label all edges in $G_1$ red
up-edges and all edges in $G_2$ blue down edges. The algorithm we used
to find partitions of the disjoint union graph into alternating cycles
consists of two layers of Depth First Search (DFS); the first layer to
find cycles and the second layer to find possible partitions into
cycles.

1.  In the first layer, using DFS, we find all alternating cycles, which
    start at a red up-edge with lower vertex $v$ and terminate with a
    blue down-edge with lower vertex $v$.

2.  In the second, in order to find partitions we first pick an
    arbitrary up-edge $e_0$. Find all cycles $C_0$ starting at $e_0$
    using DFS. For each such cycle $c_0 \in C_0$ consider the remaining
    edges not appearing in $c_0$ from $G_1, G_2$. Out of these, pick an
    arbitrary red up-edge $e_1$ and again find all cycles $C_1$ within
    the remaining edges not used in any previous cycles starting at
    $e_1$. Recurse, until no red up-edges are left remaining.

    1.  At this point, if all blue down-edges have also been used, we
        have found a partition into cycles, and we append it to a list
        of possible partitions.

    2.  If blue down-edges are left remaining, we backtrack, discarding
        the last cycle $c_k$ and continue recursion by replacing it with
        the next cycle from $C_k$. If no cycles are left in $C_k$ we
        again backtrack, discarding $c_{k-1}$ and consider the next
        cycle from $C_{k-1}$ and so on.

3.  We have found all possible partitions.

#### **Finding Minimal Partitions.** 

If two cycles share a vertex, we can plainly see that we can combine
them into a single, longer cycle. However, this longer cycle is of
little interest to us, as it is really just two shorter cycles glued
together. If we are for example interested in finding graphs that
feature length 8 cycles in their partition, we are not interested in
length 8 cycles which can be broken down into two length 4 cycles,
rather, we want to find cycles of length 8 which cannot be split apart.

Thus, we call a partition minimal if none of its cycles can be further
subdivided into shorter cycles. In our algorithm, we expand this
further: If we have two partitions into cycles $p_1, p_2$ we can
consider the lengths of cycles in $P_1$ and $P_2$ in descending order
and represent $P_1$ as $(l_{1}, l_{2},\dots,l_{k_{1}}, 0, 0, \dots)$
with $l_{i}\geq l_{i+1}$ and $P_2$ as
$(m_{1}, m_{2},\dots,m_{k_{2}}, 0, 0, \dots)$ with $m_i \geq m_{i+1}$.
We can then compare these in lexicographical order, which we denote
$\triangleleft$. We pick a partition into cycles which is minimal with
respect to this ordering to represent the colliding pair. Further work
could explore all possible partitions into cycles and look for
interesting results there; this was not our focus.

Lastly, we would like to note that we perform this \"minimum finding\"
in-place during the search and not as a secondary step, as it is vastly
more memory efficient. Here is an algorithmic description:

```r
Algorithm: Partition Pair
Require: Graphs (V, E_a), (V, E_b) on the same vertices.
Ensure: Lexicographically minimal partition P_min

# Initialization
p_min <- (infinity)    # Current minimal partition.
p <- {}                # Current partition.
U_a <- {}              # Used edges.
U_b <- {}              # Used edges.

Function Search(p, U_a, U_b):
    if U_a == E_a and U_b == E_b:
        # Found a valid partition; check minimality in-place
        if p < p_min:  # "<" denotes ordering as above.
            p_min <- p
        return

    Pick e in E_a \ U_a
    C <- Call CycleDFS(e, U_a, U_b)
    
    for c in C:
        Add c edges to U_a, U_b
        Call Search(p union {c}, U_a, U_b)
        Remove c edges from U_a, U_b
    end for
End Function

Function CycleDFS(e, U_a, U_b):
    Perform DFS recursively, alternating between unused edges from E_a and E_b starting with e.
    Return List of found cycles.
End Function
```

## Numerical Results

First of all, we would like to remind the reader of the [restriction to
non-multiset graphs](#par:restrict) that we make for our computations.
This restriction holds for all the following numerical results.
Furthermore, the proof for the following fact is \"proven by
computation\". Results were obtained using the \"Compute Collision
Sets\" feature, with the corresponding number of vertically stacked
vertices drawn, \"Ignore Dangling Vertices\" disabled and no edges
drawn.

> [!IMPORTANT]
> **Fact** 
> All pairs of vertical graphs with up to seven vertices and which share a VPHT, are colliding pairs, that is, their disjoint union graph partitions into alternating cycles, and they are of type $\mathcal{G}_1,\mathcal{G}_2$.

> [!NOTE]
> **Remark** 
> [[fact:part7]](#fact:part7) was obtained with "Common Edges Excluded from Cycle Search" (see [1.1](#app-overview)) as edges common to both always from a cycle, and thus their inclusion cannot break partitionability.

> [!IMPORTANT]
> **Corollary** 
> All pairs of vertical graphs with up to seven vertices and which share a VPHT form colliding pairs, and there exists an alternating cycle partition such that all common edges form cycles of length two.

> [!NOTE]
> **Proof**
> By [[rem:about-part7]](#rem:about-part7) and [[fact:part7]](#fact:part7) we can conclude that the disjoint union graph for all such pairs with up to seven vertices has the desired property. ◻

> [!NOTE]
> **Remark**
> This implies, that if we would remove all common edges, we could still obtain a decomposition into alternating cycles.

[^1]: Notice how the \"Compute Colliding Graphs\" brute force search is
    nothing but a search for the collision set which contains the drawn
    graph.

[^2]: []{#foot:constant label="foot:constant"}As a result of having the
    same VPHT both of these are constant within a collision set.

[^3]: How this is defined and works is document in the UI, and further
    ellaborated on in
    [1.2.4.1](#finding-minimal-partitions){reference-type="ref+label"
    reference="finding-minimal-partitions"}.

[^4]: That is, consider the two graphs where the respective edges common
    to both of them are removed.

[^5]: []{#foot:respect label="foot:respect"}Respecting the option for
    \"Ignore Dangling Vertices\".
