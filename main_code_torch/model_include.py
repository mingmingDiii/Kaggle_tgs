from main_code_torch.models.dense_dilation import DENSE_DILATION
from main_code_torch.models.dense_normal import DENSE_NORMAL
from main_code_torch.models.dpn_normal import DPN_NORMAL
from main_code_torch.models.link_dense import LINK_DENSE


models = {

    'dense_dilation':lambda :DENSE_DILATION(),
    'dense_normal': lambda: DENSE_NORMAL(),
    'dpn_normal': lambda: DPN_NORMAL(),
    'link_dense':lambda :LINK_DENSE(),


}