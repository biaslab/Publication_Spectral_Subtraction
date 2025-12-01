"""
Stream processing implementation for real-time audio.
"""
function process(::StreamProcessing, ha::AbstractHearingAid, block::SampleBuf)
    # Directly process the block in real-time
    output_signal, results = process_block(ha, block)
    return output_signal, results
end
