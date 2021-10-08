#include <R.h>
#include <Rinternals.h>

#define UNKNOWN_BRANCH_LEFT 0
#define UNKNOWN_BRANCH_RIGHT 1
#define UNKNOWN_BRANCH_RANDOM 2

#define CATEGORICAL_FEATURE 0
#define CONTINUOUS_FEATURE 1

static SEXP getListElement(SEXP list, SEXP names, const char *str)
{
    int i;
    const char *name;

    for (i = 0; i < length(list); ++i) 
    {
        name = CHAR(STRING_ELT(names, i));
        if (strcmp(name, str) == 0)
            return VECTOR_ELT(list, i);
    }
    return R_NilValue;
}


struct ct_node {

    // maximum size of set (i.e., length of dense/sparse arrays)
    int u;

    // number of observations in node
    int n;

    // majority label for observations in this node (only set for leaf nodes)
    int label;

    // dense representation of node (contains observation indices)
    int *dense;

    // name of the feature that this node handles
    char *feature;

    // categorical features are type 0; continuous features, type 1
    int feature_type;

    // splitting threshold for node (if it handles continuous features)
    // observations with values less than or equal to h get sent left
    double h;

    // splitting value for node (if it handles categorical features)
    // observations with values equal to k get sent left
    int k;

    // flag to indicate which branch handles missing data (0 sends unknowns 
    // left, 1 sends unknowns right)
    int unknown_branch;

    // splitting depth of node
    int depth;

    // loss within node
    double loss;

    struct ct_node *left;

    struct ct_node *right;

    struct ct_node *parent;

};


struct ct_node *ct_node_alloc(int u)
{
    struct ct_node *a = malloc(sizeof(*a));
    a->u = u;
    a->n = 0;
    a->h = 0;
    a->k = 0;
    a->loss = 0;
    a->unknown_branch = UNKNOWN_BRANCH_RANDOM;
    a->depth = 0;
    a->feature = NULL;
    a->feature_type = 0;
    a->dense = u > 0 ? calloc(u, sizeof(int)) : NULL;
    a->left = NULL;
    a->right = NULL;
    a->parent = NULL;
    return a;
}


// free a single node
void ct_node_free(struct ct_node *a)
{
    free(a->feature);
    free(a->dense);
    free(a);
}


// free a whole tree
static void ct_free_(struct ct_node *root)
{
    struct ct_node *p;
    struct ct_node *q;

    p = root;
    while (p)
    {
        if (p->left) {
            p = p->left;
            continue;
        } else if (p->right) {
            p = p->right;
            continue;
        } else {
            q = p->parent;
            if (q && q->left == p)
                q->left = NULL;
            else if (q && q->right == p)
                q->right = NULL;
            ct_node_free(p);
            p = q;
        }
    }
}


static int ct_node_isempty(struct ct_node *s)
{
    return s->n == 0;
}


static void ct_node_add(int i, struct ct_node *s)
{
    s->dense[s->n] = i;
    s->n += 1;
}


static void ct_node_clear(struct ct_node *s)
{
    s->n = 0;
}


static void ct_node_split_on_threshold(struct ct_node *a, struct ct_node *b, 
    struct ct_node *c, double h, double *x, int unknown_b)
{
    int i;
    for (i = 0; i < a->n; ++i)
    {
        if (x[a->dense[i]] == NA_REAL)
        {
            if (unknown_b == UNKNOWN_BRANCH_LEFT)
                ct_node_add(a->dense[i], b);
            else if (unknown_b == UNKNOWN_BRANCH_RIGHT)
                ct_node_add(a->dense[i], c);
            else
                ct_node_add(a->dense[i], unif_rand() < 0.5 ? b : c);
        }
        else
        {
            if (x[a->dense[i]] <= h)
                ct_node_add(a->dense[i], b);
            else
                ct_node_add(a->dense[i], c);
        }
    }
}


static void ct_node_split_on_value(struct ct_node *a, struct ct_node *b, 
    struct ct_node *c, int k, int *x, int unknown_b)
{
    int i;
    for (i = 0; i < a->n; ++i)
    {
        if (x[a->dense[i]] == NA_INTEGER)
        {
            if (unknown_b == UNKNOWN_BRANCH_LEFT)
                ct_node_add(a->dense[i], b);
            else if (unknown_b == UNKNOWN_BRANCH_RIGHT)
                ct_node_add(a->dense[i], c);
            else
                ct_node_add(a->dense[i], unif_rand() < 0.5 ? b : c);
        }
        else
        {
            if (x[a->dense[i]] == k)
                ct_node_add(a->dense[i], b);
            else
                ct_node_add(a->dense[i], c);
        }
    }
}


static double ct_node_loss_within(struct ct_node *a, int *labels, 
    double *weights)
{
    int i;
    int j;
    double loss = 0;

    for (i = 1; i < a->n; ++i)
    {
        for (j = 0; j < i; ++j)
        {
            // different classes but put into same group
            if (labels[a->dense[i]] != labels[a->dense[j]])
                loss += weights[a->dense[i]] + weights[a->dense[j]];
        }
    }

    return loss;
}


static double ct_node_loss_between(struct ct_node *a, struct ct_node *b, 
    int *labels, double *weights)
{
    int i;
    int j;
    double loss = 0;

    for (i = 0; i < a->n; ++i)
    {
        for (j = 0; j < b->n; ++j)
        {
            // same class but split into two groups
            if (labels[a->dense[i]] == labels[b->dense[j]])
                loss += weights[a->dense[i]] + weights[b->dense[j]];
        }
    }

    return loss;
}


static int ct_node_maybe_split(struct ct_node *a, int *labels, 
    double *weights, int max_depth, double stopping_tol)
{
    a->loss = ct_node_loss_within(a, labels, weights);
    if (max_depth != NA_INTEGER) {
        if (a->depth < max_depth && a->loss > stopping_tol)
            return 1;
        return 0;
    }
    if (a->loss > stopping_tol)
        return 1;
    return 0;
}


static void ct_node_split_categorical(
    struct ct_node **a,
    struct ct_node **b,
    struct ct_node **c,
    int feature_index,
    SEXP feature,
    int *labels, 
    double *weights,
    int ub_l,
    int ub_u,
    int *best_feature_index,
    int *best_feature_value,
    int *best_unknown_branch,
    double *best_loss
)
{
    int j;
    double loss;
    int nlevels = length(getAttrib(feature, install("levels")));
    int *x = INTEGER(feature);
    int unknown_b;

    for (j = 1; j <= nlevels; ++j)
    {
        for (unknown_b = ub_l; unknown_b <= ub_u; ++unknown_b)
        {
            ct_node_split_on_value(*a, *b, *c, j, x, unknown_b);
            if ((*b)->n == 0 || (*c)->n == 0)
            {
                ct_node_clear(*b);
                ct_node_clear(*c);
                continue;
            }
            loss = ct_node_loss_within(*b, labels, weights) +
                ct_node_loss_within(*c, labels, weights) +
                ct_node_loss_between(*b, *c, labels, weights);
            if (loss < *best_loss)
            {
                *best_feature_index = feature_index;
                *best_feature_value = j;
                *best_loss = loss;
                *best_unknown_branch = unknown_b;
            }
            ct_node_clear(*b);
            ct_node_clear(*c);
        }
    }
}


static void ct_node_split_continuous(
    struct ct_node **a,
    struct ct_node **b,
    struct ct_node **c,
    int feature_index,
    SEXP feature,
    int *labels, 
    double *weights,
    int ub_l,
    int ub_u,
    int *best_feature_index,
    double *best_feature_threshold,
    int *best_unknown_branch,
    double *best_loss
)
{
    int j;
    double loss;
    double *x = REAL(feature);
    int unknown_b;

    for (j = 0; j < (*a)->n; ++j)
    {
        if (x[j] == NA_REAL)
            continue;
        for (unknown_b = ub_l; unknown_b <= ub_u; ++unknown_b)
        {
            ct_node_split_on_threshold(*a, *b, *c, x[j], x, unknown_b);
            if ((*b)->n == 0 || (*c)->n == 0)
            {
                ct_node_clear(*b);
                ct_node_clear(*c);
                continue;
            }
            loss = ct_node_loss_within(*b, labels, weights) +
                ct_node_loss_within(*c, labels, weights) +
                ct_node_loss_between(*b, *c, labels, weights);
            if (loss < *best_loss)
            {
                *best_feature_index = feature_index;
                *best_feature_threshold = x[j];
                *best_loss = loss;
                *best_unknown_branch = unknown_b;
            }
            ct_node_clear(*b);
            ct_node_clear(*c);
        }
    }
}


static int ct_node_best_split(
    struct ct_node *a,
    SEXP features,
    int *labels, 
    double *weights, 
    int train_unknown
)
{
    int i;
    int p = length(features);
    int ub_l;
    int ub_u;

    if (train_unknown)
    {
        ub_l = UNKNOWN_BRANCH_LEFT;
        ub_u = UNKNOWN_BRANCH_RIGHT;
    }
    else
    {
        ub_l = UNKNOWN_BRANCH_RANDOM;
        ub_u = UNKNOWN_BRANCH_RANDOM;
    }

    double loss;

    SEXP feature;
    SEXP feature_name;

    struct ct_node *b = ct_node_alloc(a->n);
    struct ct_node *c = ct_node_alloc(a->n);

    int best_feature_index = -1;
    int best_feature_value;
    int best_unknown_branch = UNKNOWN_BRANCH_RANDOM;
    double best_feature_threshold;
    double best_loss = R_PosInf;

    for (i = 0; i < p; ++i)
    {
        feature = PROTECT(VECTOR_ELT(features, i));
        if (TYPEOF(feature) == INTSXP)
            ct_node_split_categorical(
                &a, &b, &c, i, feature, labels, weights, ub_l, ub_u,  
                &best_feature_index, &best_feature_value, &best_unknown_branch, 
                &best_loss);
        else
            ct_node_split_continuous(
                &a, &b, &c, i, feature, labels, weights, ub_l, ub_u, 
                &best_feature_index, &best_feature_threshold, 
                &best_unknown_branch, &best_loss);
        
        UNPROTECT(1);
    }

    if (best_feature_index < 0)
        goto split_failed;
    
    feature = PROTECT(VECTOR_ELT(features, best_feature_index));
    feature_name = PROTECT(STRING_ELT(
        getAttrib(features, R_NamesSymbol), best_feature_index));

    int nchar = strlen(CHAR(feature_name));
    char *name = calloc(nchar+1, sizeof(*name));

    strncpy(name, CHAR(feature_name), nchar);

    if (TYPEOF(feature) == INTSXP)
    {
        ct_node_split_on_value(a, b, c, best_feature_value, INTEGER(feature), 
            best_unknown_branch);
        a->k = best_feature_value;
        a->feature_type = CATEGORICAL_FEATURE;
    }
    else
    {
        ct_node_split_on_threshold(a, b, c, best_feature_threshold, REAL(feature), 
            best_unknown_branch);
        a->h = best_feature_threshold;
        a->feature_type = CONTINUOUS_FEATURE;
    }

    a->unknown_branch = best_unknown_branch;
    a->feature = name;
    a->left = b;
    a->right = c;
    b->parent = a;
    c->parent = a;
    b->depth = a->depth + 1;
    c->depth = a->depth + 1;
    UNPROTECT(2);
    return 0;

    split_failed:
        ct_node_free(b);
        ct_node_free(c);
        return 1;
}


static int ct_predict(struct ct_node *root, SEXP obs)
{
    SEXP feature;
    SEXP feature_names = PROTECT(getAttrib(obs, R_NamesSymbol));

    struct ct_node *p = root;

    while (p->left && p->right)
    {
        feature = PROTECT(getListElement(obs, feature_names, p->feature));
        if (feature == R_NilValue)
        {
            if (p->unknown_branch == UNKNOWN_BRANCH_LEFT)
                p = p->left;
            else if (p->unknown_branch == UNKNOWN_BRANCH_RIGHT)
                p = p->right;
            else
                p = unif_rand() < 0.5 ? p->left : p->right;
        }
        else
        {
            if (TYPEOF(feature) == INTSXP)
            {
                if (INTEGER(feature)[0] == NA_INTEGER)
                {
                    if (p->unknown_branch == UNKNOWN_BRANCH_LEFT)
                        p = p->left;
                    else if (p->unknown_branch == UNKNOWN_BRANCH_RIGHT)
                        p = p->right;
                    else 
                        p = unif_rand() < 0.5 ? p->left : p->right;
                }
                else
                {
                    if (INTEGER(feature)[0] == p->k)
                        p = p->left;
                    else
                        p = p->right;
                }
            }
            else
            {
                if (REAL(feature)[0] == NA_REAL)
                {
                    if (p->unknown_branch == UNKNOWN_BRANCH_LEFT)
                        p = p->left;
                    else if (p->unknown_branch == UNKNOWN_BRANCH_RIGHT)
                        p = p->right;
                    else
                        p = unif_rand() < 0.5 ? p->left : p->right;
                }
                else
                {
                    if (REAL(feature)[0] <= p->h)
                        p = p->left;
                    else
                        p = p->right;
                }
            }
        }
        UNPROTECT(1);
    }
    UNPROTECT(1);
    return p->label;
}


static double ct_prediction_error(
    struct ct_node *root, 
    int *labels,
    double *weights,
    SEXP features
)
{
    int i;
    int j;
    int nfeatures = length(features);
    int nobs = length(VECTOR_ELT(features, 0));

    double error = 0;

    SEXP obs = PROTECT(allocVector(VECSXP, nfeatures));
    setAttrib(obs, R_NamesSymbol, getAttrib(features, R_NamesSymbol));

    for (i = 0; i < nobs; ++i)
    {
        for (j = 0; j < nfeatures; ++j)
        {
            switch (TYPEOF(VECTOR_ELT(features, j)))
            {
                case REALSXP:
                    SET_VECTOR_ELT(obs, j, ScalarReal(
                        REAL(VECTOR_ELT(features, j))[i]));
                    break;
                case INTSXP:
                    SET_VECTOR_ELT(obs, j, ScalarInteger(
                        INTEGER(VECTOR_ELT(features, j))[i]));
                    break;
            }
        }
        if (ct_predict(root, obs) != labels[i])
            error += weights[i];
    }

    UNPROTECT(1);
    return error;
}


static void build_ct_recurse(
    struct ct_node *a,
    SEXP features,
    int *labels,
    double *weights,
    int nlabels,
    int train_unknown,
    int max_depth,
    double stopping_tol
)
{
    int i;
    int majority_label;
    int counts[nlabels];
    if (ct_node_maybe_split(a, labels, weights, max_depth, stopping_tol))
    {
        //Rprintf("%d - here1\n", a->n);
        if (ct_node_best_split(a, features, labels, weights, train_unknown))
            goto split_failed;
        //Rprintf("here2\n");
        build_ct_recurse(
            a->left, features, labels, weights, nlabels, train_unknown, 
            max_depth, stopping_tol);
        build_ct_recurse(
            a->right, features, labels, weights, nlabels, train_unknown, 
            max_depth, stopping_tol);
    }
    else
    {
        split_failed:
            majority_label = 1;

            memset(counts, 0, nlabels * sizeof(int));

            for (i = 0; i < a->n; ++i)
                counts[labels[a->dense[i]]-1] += 1; // labels start at 1

            for (i = 1; i < nlabels; ++i)
            {
                if (counts[i] > counts[majority_label-1])
                    majority_label = i+1;
            }

            a->label = majority_label;
    }
}


static struct ct_node *build_ct(
    SEXP features,
    SEXP labels,
    SEXP weights,
    SEXP control,
    double *error
)
{
    int i;
    int u = length(labels);
    int nlabels = length(getAttrib(labels, install("levels")));

    struct ct_node *root;

    SEXP max_depth;
    SEXP train_unknown;
    SEXP stopping_tol;
    SEXP names = PROTECT(getAttrib(control, R_NamesSymbol));

    max_depth = getListElement(control, names, "max_depth");
    stopping_tol = getListElement(control, names, "stopping_tol");
    train_unknown = getListElement(control, names, "train_unknown");

    root = ct_node_alloc(u);
    for (i = 0; i < u; ++i)
        ct_node_add(i, root);
    
    build_ct_recurse(
        root,
        features, 
        INTEGER(labels), 
        REAL(weights),
        nlabels, 
        INTEGER(train_unknown)[0],
        INTEGER(max_depth)[0],
        REAL(stopping_tol)[0]);
    
    *error = ct_prediction_error(
        root, INTEGER(labels), REAL(weights), features);

    UNPROTECT(1);
    return root;
}


static void ct_node_print(struct ct_node *a, int depth)
{
    int w;
    if (a->parent)
    {
        w = strlen(a->parent->feature) + depth;
        if (a->parent->feature_type == CATEGORICAL_FEATURE)
        {
            if (a == a->parent->left)
                Rprintf("%*s == %d (loss: %f)\n", w, a->parent->feature, 
                    a->parent->k, a->loss);
            else
                Rprintf("%*s != %d (loss: %f)\n", w, a->parent->feature, 
                    a->parent->k, a->loss);
        }
        else
        {
            if (a == a->parent->left)
                Rprintf("%*s <= %f (loss: %f)\n", w, a->parent->feature, 
                    a->parent->h, a->loss);
            else
                Rprintf("%*s > %f (loss: %f)\n", w, a->parent->feature, 
                    a->parent->h, a->loss);
        }
        if (a->left == 0 && a->right == 0)
        {
            Rprintf("%*s", depth+1+3, "-->");
            Rprintf(" %d\n", a->label);
        }
    }
    else
    {
        Rprintf("ROOT (loss: %f)\n", a->loss);
    }
}


static void ct_node_print_recurse(struct ct_node *a, int depth)
{
    ct_node_print(a, depth);
    if (a->left && a->right)
    {
        ct_node_print_recurse(a->left, depth+1);
        ct_node_print_recurse(a->right, depth+1);
    }
}

static void ct_print(struct ct_node *root)
{
    ct_node_print_recurse(root, 0);
}


void ct_free(SEXP e)
{
    struct ct_node *root = (struct ct_node *)R_ExternalPtrAddr(e);
    ct_free_(root);
    R_ClearExternalPtr(e);
}


SEXP C_ct_fit(
    SEXP features,
    SEXP labels,
    SEXP weights,
    SEXP control
)
{
    double prediction_error;
    struct ct_node *tree;
    SEXP exptr;

    tree = build_ct(features, labels, weights, control, &prediction_error);

    exptr = PROTECT(R_MakeExternalPtr(tree, R_NilValue, R_NilValue));

    setAttrib(exptr, install("error"), ScalarReal(prediction_error));

    R_RegisterCFinalizer(exptr, ct_free);

    UNPROTECT(1);
    return exptr;
}


SEXP C_ct_predict(SEXP exptr, SEXP observations)
{
    int i;
    int j;
    int nfeatures = length(observations);
    int nobs = length(VECTOR_ELT(observations, 0));

    struct ct_node *root = (struct ct_node *)R_ExternalPtrAddr(exptr);

    SEXP p = PROTECT(allocVector(INTSXP, nobs));
    SEXP obs = PROTECT(allocVector(VECSXP, nfeatures));
    setAttrib(obs, R_NamesSymbol, getAttrib(observations, R_NamesSymbol));

    GetRNGstate();

    for (i = 0; i < nobs; ++i)
    {
        for (j = 0; j < nfeatures; ++j)
        {
            switch (TYPEOF(VECTOR_ELT(observations, j)))
            {
                case REALSXP:
                    SET_VECTOR_ELT(obs, j, ScalarReal(
                        REAL(VECTOR_ELT(observations, j))[i]));
                    break;
                case INTSXP:
                    SET_VECTOR_ELT(obs, j, ScalarInteger(
                        INTEGER(VECTOR_ELT(observations, j))[i]));
                    break;
                default:;
            }
        }
        INTEGER(p)[i] = ct_predict(root, obs);
    }

    PutRNGstate();

    UNPROTECT(2);
    return p;
}


SEXP C_ct_print(SEXP exptr)
{
    ct_print((struct ct_node *)R_ExternalPtrAddr(exptr));
    return R_NilValue;
}


static void ct_node_serialize_recurse(struct ct_node *a, SEXP s)
{
    if (a->left && a->right) 
    {
        ct_node_serialize_recurse(a->left, 
            SETCAR(s, CONS(R_NilValue, R_NilValue)));

        ct_node_serialize_recurse(a->right, 
            SETCDR(s, CONS(R_NilValue, R_NilValue)));
    }
    // serialized classification trees are only ever used for prediction,
    // which means we only need the following members of struct ct_node:
    //   label, feature, feature_type, h, k, unknown_branch, depth, loss
    SEXP data = PROTECT(allocVector(VECSXP, 8));
    SET_VECTOR_ELT(data, 0, ScalarInteger(a->label));
    SET_VECTOR_ELT(data, 1, a->feature ? mkString(a->feature) : R_NilValue);
    SET_VECTOR_ELT(data, 2, ScalarInteger(a->feature_type));
    SET_VECTOR_ELT(data, 3, ScalarReal(a->h));
    SET_VECTOR_ELT(data, 4, ScalarInteger(a->k));
    SET_VECTOR_ELT(data, 5, ScalarInteger(a->unknown_branch));
    SET_VECTOR_ELT(data, 6, ScalarInteger(a->depth));
    SET_VECTOR_ELT(data, 7, ScalarReal(a->loss));
    SET_TAG(s, data);
    UNPROTECT(1);
}


SEXP C_ct_serialize(SEXP root)
{
    SEXP s = PROTECT(CONS(R_NilValue, R_NilValue));
    ct_node_serialize_recurse(R_ExternalPtrAddr(root), s);
    UNPROTECT(1);
    return s;
}


static void ct_node_unserialize_recurse(struct ct_node *a, SEXP s)
{
    if (CAR(s) != R_NilValue && CDR(s) != R_NilValue)
    {
        a->left = ct_node_alloc(0);
        a->left->parent = a;
        ct_node_unserialize_recurse(a->left, CAR(s));
        a->right = ct_node_alloc(0);
        a->right->parent = a;
        ct_node_unserialize_recurse(a->right, CDR(s));
    }

    SEXP data = PROTECT(TAG(s));
    
    if (VECTOR_ELT(data, 1) != R_NilValue)
    {
        const char *name = CHAR(STRING_ELT(VECTOR_ELT(data, 1), 0));
        a->feature = calloc(strlen(name)+1, sizeof(*name));
        strncpy(a->feature, name, strlen(name));
    }

    a->label = INTEGER(VECTOR_ELT(data, 0))[0];
    a->feature_type = INTEGER(VECTOR_ELT(data, 2))[0];
    a->h = REAL(VECTOR_ELT(data, 3))[0];
    a->k = INTEGER(VECTOR_ELT(data, 4))[0];
    a->unknown_branch = INTEGER(VECTOR_ELT(data, 5))[0];
    a->depth = INTEGER(VECTOR_ELT(data, 6))[0];
    a->loss = REAL(VECTOR_ELT(data, 7))[0];
    UNPROTECT(1);
}


SEXP C_ct_unserialize(SEXP s)
{
    struct ct_node *root = ct_node_alloc(0);
    ct_node_unserialize_recurse(root, s);
    return R_MakeExternalPtr(root, R_NilValue, R_NilValue);
}


#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

#define CALLDEF(name, n)  {#name, (DL_FUNC) &name, n}

SEXP C_ct_fit(SEXP, SEXP, SEXP, SEXP);
SEXP C_ct_predict(SEXP, SEXP);
SEXP C_ct_print(SEXP);
SEXP C_ct_serialize(SEXP);
SEXP C_ct_unserialize(SEXP);


void attribute_visible R_init_ct(DllInfo *info)
{
    R_registerRoutines(info, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
    R_forceSymbols(info, TRUE);
}

