"""r1 prototype: a third-party output adapter for sklearn's set_output.

Implements sklearn's 4-method ContainerAdapterProtocol for pyarrow.Table, to test
how open the dispatcher really is (it is private API: sklearn.utils._set_output).
"""

import numpy as np


class PyArrowAdapter:
    container_lib = "pyarrow"

    def create_container(self, X_output, X_original, columns, inplace=True):
        import pyarrow as pa
        from sklearn.utils._set_output import get_columns

        columns = get_columns(columns)
        if isinstance(X_output, pa.Table):
            return X_output if columns is None else self.rename_columns(X_output, columns)
        arr = np.asarray(X_output)
        if columns is None:
            columns = [f"x{i}" for i in range(arr.shape[1])]
        return pa.table({str(c): arr[:, i] for i, c in enumerate(columns)})

    def is_supported_container(self, X):
        import pyarrow as pa

        return isinstance(X, pa.Table)

    def rename_columns(self, X, columns):
        return X.rename_columns([str(c) for c in columns])

    def hstack(self, Xs):
        import pyarrow as pa

        cols, names = [], []
        for t in Xs:
            cols.extend(t.columns)
            names.extend(t.column_names)
        return pa.table(cols, names=names)


def register():
    from sklearn.utils._set_output import ADAPTERS_MANAGER

    ADAPTERS_MANAGER.register(PyArrowAdapter())
