#' Classification tree learning of taxon concepts
#'
#' @param x A data.frame. Rows represent individual specimens and columns
#' represent continuous and categorical characters measured/observed for
#' each specimen.
#' @param y A factor vector mapping rows in \code{x} to a set of taxon
#' concepts.
#' @param weights A numeric vector assigning weights to each observation in
#' \code{x}. Optional. If omitted each observation is assigned equal weight.
#' @param control A list of options for the learning algorithm:
#' \describe{
#' \item{max_depth}{The maximum depth of the classification tree. The default
#' value of \code{NA} grows trees until each leaf is pure.}
#' \item{stopping_tol}{The maximum value of loss allowed within a node before
#' it is considered pure.}
#' \item{train_unknown}{A boolean. Should the learning algorithm optimize the
#' placement of features with missing data?}
#' }
#' @return A classification tree of \class{ct} that can be used to predict
#' the taxon concept of a specimen.
#' @examples
#' x = iris[, -5]
#' y = iris[, 5]
#' fit = ct(x, y)
#' plot(x, col=y)
#' plot(x, col=predict(fit, x))
ct_fit = function(x, y, weights, control=list(
    max_depth=NA, stopping_tol=0, train_unknown=anyNA(x))) {

    if (missing(weights))
        weights = rep(1 / nrow(x), nrow(x))

    stopifnot(is.data.frame(x))
    stopifnot(is.factor(y))
    stopifnot(is.numeric(weights))

    col_classes = unlist(lapply(x, class))

    stopifnot(all(col_classes %in% c("numeric", "factor")))

    if (is.null(control$max_depth)) {
        control$max_depth = NA_integer_
    } else {
        if (is.na(control$max_depth))
            control$max_depth = NA_integer_
        else if (as.integer(control$max_depth) <= 0)
            stop(gettextf("max_depth must be greater than 0"))
        else
            control$max_depth = as.integer(control$max_depth)
    }
    
    if (is.null(control$stopping_tol) || is.na(control$stopping_tol))
        control$stopping_tol = 0
    else if (control$stopping_tol < 0 || !is.numeric(control$stopping_tol))
        stop(gettextf("stopping_tol must be greater than 0"))

    if (is.null(control$train_unknown) || is.na(control$train_unknown))
        control$train_unknown = anyNA(x)
    else if (!is.logical(control$train_unknown))
        stop(gettextf("train_unknown must be TRUE or FALSE"))
    

    structure(.Call(C_ct_fit, x, y, weights / sum(weights), control), 
        class="ct", levels=levels(y))

}


predict.ct = function(object, x, ...) {
    stopifnot(is.data.frame(x))
    col_classes = unlist(lapply(x, class))
    stopifnot(all(col_classes %in% c("numeric", "factor")))

    factor(attr(object, "levels")[.Call(C_ct_predict, object, x)])
}


print.ct = function(x, ...) {
    invisible(.Call(C_ct_print, x))
}


#' Boosted classification tree learning of taxon concepts
#'
#' @param x A data.frame. Rows represent individual specimens and columns
#' represent continuous and categorical characters measured/observed for
#' each specimen.
#' @param y A factor vector mapping rows in \code{x} to a set of taxon
#' concepts.
#' @param niter The number of boosting iterations to perform.
#' @param control A list of options for the learning algorithm.
#' @return An ensemble of classification trees of \class{ct_boost} that can be 
#' used to predict the taxon concept of a specimen.
#' @seealso \link{\code{ct}}
#' @examples
#' x = iris[, -5]
#' y = iris[, 5]
#' fit = boost_ct(x, y, niter=10, control=list(max_depth=2))
#' plot(x, col=y)
#' plot(x, col=predict(fit, x))
boost_ct = function(x, y, niter=10L, control=list(
    max_depth=NA, stopping_tol=0, train_unknown=anyNA(x))) 
{
    stopifnot(is.data.frame(x))
    stopifnot(is.factor(y))

    ensemble = vector("list", niter)
    a = numeric(niter)
    w = rep(1 / nrow(x), nrow(x))

    labels = match(y, levels(y))
    K = nlevels(y)

    for (i in 1:niter) {
        fit = ct(x, y, w, control)
        e = attr(fit, "error")
        a[i] = log((1 - e) / e) + log(K - 1)

        w = w * exp(a[i] * (predict(fit, x) != labels))
        w = w / sum(w)

        ensemble[[i]] = fit
    }
    structure(list(a, ensemble), class="ct_boost")
}

predict.ct_boost = function(object, x, ...) {
    stopifnot(is.data.frame(x))
    col_classes = unlist(lapply(x, class))
    stopifnot(all(col_classes %in% c("numeric", "factor")))

    a = object[[1]]
    ensemble = object[[2]]
    n = nrow(x)
    K = length(attr(ensemble[[1]], "levels"))

    p = matrix(0, n, K)

    row_index = 1:n

    for (i in 1:length(ensemble)) {
        col_index = as.integer(predict(ensemble[[i]], x))

        flat_index = (row_index-1) + (col_index-1) * n + 1

        p[flat_index] = p[flat_index] + a[i]
        
    }

    factor(attr(ensemble[[1]], "levels")[apply(p, 1, which.max)])
}


serialize_ct = function(object, connection) {
    stopifnot(inherits(object, "ct"))
    s = .Call(C_ct_serialize, object)
    attr(s, "levels") = attr(object, "levels")
    serialize(s, connection)
}


unserialize_ct = function(connection) {
    obj = unserialize(connection)
    structure(.Call(C_ct_unserialize, obj), class="ct", 
        levels=attr(obj, "levels"))
}


serialize_ct_boost = function(object, connection) {
    stopifnot(inherits(object, "ct_boost"))

    for (i in 1:length(object[[2]]))
        object[[2]][[i]] = serialize_ct(object[[2]][[i]], NULL)

    serialize(object, connection)
}


unserialize_ct_boost = function(connection) {
    object = unserialize(connection)
    for (i in 1:length(object[[2]]))
        object[[2]][[i]] = unserialize_ct(object[[2]][[i]])
    object
}