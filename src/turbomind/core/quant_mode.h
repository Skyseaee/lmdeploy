#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace turbomind {

class QuantMode {
public:
    using BaseType = std::uint32_t;

    constexpr explicit QuantMode(BaseType value = 0) noexcept: mValue{value} {}

    static constexpr QuantMode int4Weights() noexcept
    {
        return QuantMode(1u << 0);
    }
    static constexpr QuantMode int8Weights() noexcept
    {
        return QuantMode(1u << 1);
    }
    static constexpr QuantMode activations() noexcept
    {
        return QuantMode(1u << 2);
    }
    static constexpr QuantMode perGroupScaling() noexcept
    {
        return QuantMode(1u << 5);
    }
    static constexpr QuantMode fp8Qdq() noexcept
    {
        return QuantMode(1u << 8);
    }
    static constexpr QuantMode fp8BlockScales() noexcept
    {
        return QuantMode(1u << 10);
    }
    static constexpr QuantMode nvfp4() noexcept
    {
        return QuantMode(1u << 12);
    }

    static constexpr QuantMode fromDescription()
    {
        QuantMode quantMode{};
        return quantMode;
    }

    static QuantMode fromQuantAlgo(const std::string& quantAlgo)
    {
        QuantMode quantMode;
        if (quantAlgo == "w4a16_awq") {
            quantMode += int4Weights();
            quantMode += perGroupScaling();
        }
        else if (quantAlgo == "w4a8_awq") {
            quantMode += int4Weights();
            quantMode += perGroupScaling();
            quantMode += activations();
        }
        else if (quantAlgo == "w8a8_sq") {
            quantMode += int8Weights();
            quantMode += activations();
        }
        else if (quantAlgo == "fp8_static") {
            quantMode += fp8Qdq();
            quantMode += activations();
        }
        else if (quantAlgo == "fp8_block_scales") {
            quantMode += fp8BlockScales();
        }
        else if (quantAlgo == "fp4") {
            quantMode += nvfp4();
        }
        return quantMode;
    }

    constexpr bool isSet(const QuantMode& mode) const noexcept
    {
        return (mValue & mode.mValue) == mode.mValue;
    }

    constexpr bool isW4A16AWQ() const noexcept
    {
        return isSet(int4Weights() + perGroupScaling()) && !isSet(activations());
    }

    constexpr bool isW4A8AWQ() const noexcept
    {
        return isSet(int4Weights() + perGroupScaling() + activations());
    }

    constexpr bool isW8A8SQPerTensor() const noexcept
    {
        return isSet(int8Weights() + activations()) && !isSet(perGroupScaling());
    }

    constexpr bool isFP8Static() const noexcept
    {
        return isSet(fp8Qdq() + activations());
    }

    constexpr bool isFP8BlockScales() const noexcept
    {
        return isSet(fp8BlockScales());
    }

    constexpr bool isFP4() const noexcept
    {
        return isSet(nvfp4());
    }

    constexpr QuantMode operator+(QuantMode const& other) const noexcept
    {
        return QuantMode(mValue | other.mValue);
    }

    constexpr QuantMode& operator+=(QuantMode const& other) noexcept
    {
        return *this = *this + other;
    }

    constexpr QuantMode operator-(QuantMode const& other) const noexcept
    {
        return QuantMode(mValue & ~other.mValue);
    }

    constexpr QuantMode& operator-=(QuantMode const& other) noexcept
    {
        return *this = *this - other;
    }

    constexpr bool operator==(QuantMode const& other) const noexcept
    {
        return mValue == other.mValue;
    }

    constexpr bool operator!=(QuantMode const& other) const noexcept
    {
        return !(*this == other);
    }

    constexpr BaseType value() const noexcept
    {
        return mValue;
    }

    std::string to_string() const
    {
        if (isSet(fp8Qdq()) && isSet(activations()) && !isSet(fp8BlockScales())) {
            return "fp8_static";
        }
        if (isSet(fp8BlockScales())) {
            return "fp8_block_scales";
        }
        if (isSet(int4Weights()) && isSet(perGroupScaling()) && !isSet(activations())) {
            return "w4a16_awq";
        }
        if (isSet(int4Weights()) && isSet(perGroupScaling()) && isSet(activations())) {
            return "w4a8_awq";
        }
        if (isSet(int8Weights()) && isSet(activations())) {
            return "w8a8_sq";
        }
        if (isSet(nvfp4())) {
            return "fp4";
        }
        return "unknown";
    }

private:
    BaseType mValue;
};

}  // namespace turbomind