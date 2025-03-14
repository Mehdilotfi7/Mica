using FirstPkg
using Test

@testset "FirstPkg.jl" begin
    @test FirstPkg.Functions.greet_your_package_name() == "Hello FirstPkg"
    @test FirstPkg.Functions.greet_your_package_name() != "Hello world!"
end
