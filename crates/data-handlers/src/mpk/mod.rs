
use std::{
    error::Error,
    fmt::Display,
    fs::{File},
    path::PathBuf,
};
use std::io::{BufReader, BufWriter, Write};
use memmap2::{Advice, Mmap};
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;
use crate::data_handler::{DataReader, DataWriter};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MessagePack Writer
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug)]
pub struct MpkDataWriter {
    writer: BufWriter<File>,
    file_base: String,
    record_offsets: Vec<usize>,
    offset: usize,
}

impl MpkDataWriter {
    pub fn new(file_base: &str) -> Self {
        let data_name = format!("{}-data.mpk", file_base);
        let file = File::create(&data_name).expect(&format!("Error creating data file {}", &data_name));

        Self {
            writer: BufWriter::with_capacity(64 * 1_024, file),
            file_base: file_base.to_string(),
            record_offsets: Vec::new(),
            offset: 0,
        }
    }
}

impl<T> DataWriter<T> for MpkDataWriter
where T: Serialize
{
    fn write(&mut self, data: &Vec<T>) -> Result<(), Box<dyn Error>> {
        self.record_offsets.push(self.offset);
        let data = rmp_serde::to_vec(&data)?;
        self.writer.write_all(&data)?;
        self.offset += data.len();
        Ok(())
    }
}

impl Drop for MpkDataWriter {
    // on drop we will write the index file
    fn drop(&mut self) {
        self.writer.flush().unwrap();
        let index_name = format!("{}-index.mpk", &self.file_base);
        let index_file = File::create(&index_name).expect(&format!("Couldn't open output file {}", &index_name));
        let mut writer = BufWriter::with_capacity(64 * 1_024, &index_file);
        rmp_serde::encode::write(&mut writer, &self.record_offsets).expect(&format!("Error writing data to file {}", &index_name));
    }
}






//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MessagePack Reader
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug)]
pub struct MpkDataReader {
    mapped_file: Mmap,
    record_offsets: Vec<usize>,
}

impl MpkDataReader {
    pub fn new(file_base: &str) -> Self {
        let data_name = format!("{}-data.mpk", file_base);
        let index_name = format!("{}-index.mpk", file_base);

        // read in the record offsets
        let file = File::open(&index_name).expect(&format!("Couldn't open data file {:?}", &index_name));
        let reader = BufReader::with_capacity(1_024 * 1_024, &file);
        let mut record_offsets: Vec<usize> = rmp_serde::decode::from_read(reader).expect(&format!("Error reading data from file {:?}", &index_name));

        // add additional offset, being the end of the file
        let metadata = std::fs::metadata(&index_name).expect(&format!("Couldn't get metadata from file {:?}", &index_name));
        record_offsets.push(metadata.len() as usize);


        let file = File::open(&data_name).expect(&format!("Couldn't open data file {:?}", &data_name));
        let mapped_file = unsafe {
            Mmap::map(&file).expect("failed to memory map file")
        };
        mapped_file.advise(Advice::Sequential).expect("failed to advise mmap of sequential");

        Self {
            record_offsets,
            mapped_file,
        }
    }
}

impl<T> DataReader<T> for MpkDataReader
where T: DeserializeOwned
{
    fn read(&self, index: usize) -> Result<Vec<T>, Box<dyn Error>> {
        if index >= self.record_offsets.len() {
            return Err("Index out of bounds".into());
        }

        let start = self.record_offsets[index];
        let end = self.record_offsets[index + 1];

        // Slice the mmap (looks like reading from RAM)
        let bytes = &self.mapped_file[start..end];
        let data: Vec<T> = rmp_serde::from_slice(bytes)?;

        Ok(data)
    }

    fn len(&self) -> usize {
        self.record_offsets.len() - 1
    }
}






