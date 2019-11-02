from typing import List
from bundle import Bundle
from abc import ABCMeta, abstractmethod
from pypianoroll import Multitrack, Track

class Processor(metaclass=ABCMeta):
    """
    事前処理の1単位はこのクラスを継承すること．
    """
    def __init__(self, **kwargs):
        """
        処理に用い値をコンストラクタで受け取る．
        """
        self.kwargs = kwargs
    
    @abstractmethod
    def __call__(self, **args):
        """
        データを受け取り，処理を行ったデータを返す．
        実装強制．
        """
        raise NotImplementedError()
        
class BundlesProcessor(Processor, metaclass=ABCMeta):
    """
    Bundleのリストを処理するプロセッサ
    このクラスを継承したプロセッサはprocess_bundleを実装すること
    """
    @abstractmethod
    def process_bundle(self, bundle: Bundle) -> Bundle:
        raise NotImplementedError()
    
    def __call__(self, bundles: List[Bundle]) -> List[Bundle]:
        results = []
        
        for bundle in bundles:
            result = self.process_bundle(bundle)
            if result is not None:
                results.append(result)

        return results

class SequentialBundlesProcessor(Processor):
    """
    複数のBundlesProcessorによる処理をまとめて行う
    """
    def __init__(self, processors: List[BundlesProcessor]):
        self.processors = processors

    def __call__(self, bundles: List[Bundle]) -> List[Bundle]:
        for processor in self.processors:
            bundles = processor(bundles)
        return bundles
