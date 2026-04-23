class SaveIntegral[IntegralReturn]:
    def _SaveIntegral_Energy(self) -> IntegralReturn: ...
    def _SaveIntegral_Area(self) -> IntegralReturn: ...

    @staticmethod
    def Energy[T](object: 'SaveIntegral[T]') -> T:
        return object._SaveIntegral_Energy()

    @staticmethod
    def Area[T](object: 'SaveIntegral[T]') -> T:
        return object._SaveIntegral_Area()
