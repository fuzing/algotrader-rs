
use std::{
    error::Error,
    fs::{File},
};
use std::io::{BufReader, BufWriter, Write};
use std::marker::PhantomData;
use memmap2::{Advice, Mmap, MmapOptions};
use serde::{
    Serialize,
    de::DeserializeOwned
};
use crate::data_handler::{DataReader, DataWriter};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MessagePack Writer
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug)]
pub struct MpkDataWriter<T> {
    writer: BufWriter<File>,
    file_base: String,
    record_offsets: Vec<usize>,
    offset: usize,
    _marker: PhantomData<T>,
}

impl<T> MpkDataWriter<T> {
    pub fn new(file_base: &str) -> Self {
        let data_name = format!("{}.dat", file_base);
        let file = File::create(&data_name).expect(&format!("Error creating data file {}", &data_name));

        Self {
            writer: BufWriter::with_capacity(25 * 1_024 * 1_024, file),
            file_base: file_base.to_string(),
            record_offsets: Vec::new(),
            offset: 0,
            _marker: PhantomData,
        }
    }
}

impl<T> DataWriter<T> for MpkDataWriter<T>
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

impl<T> Drop for MpkDataWriter<T> {
    // on drop we will write the index file
    fn drop(&mut self) {
        self.writer.flush().unwrap();
        let index_name = format!("{}.idx", &self.file_base);
        let index_file = File::create(&index_name).expect(&format!("Couldn't open output file {}", &index_name));
        let mut writer = BufWriter::with_capacity(10 * 1_024 * 1_024, &index_file);
        rmp_serde::encode::write(&mut writer, &self.record_offsets).expect(&format!("Error writing data to file {}", &index_name));
        writer.flush().unwrap();
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MessagePack Memory Mapped Reader
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug)]
pub struct MpkDataReader<T> {
    mapped_file: Mmap,
    record_offsets: Vec<usize>,
    _marker: PhantomData<T>,
}

#[derive(Debug)]
pub enum AccessType {
    Sequential,
    Random,
}


impl<T> MpkDataReader<T> {
    pub fn new(file_base: &str, access_type: AccessType) -> Self {
        let data_name = format!("{}.dat", file_base);
        let index_name = format!("{}.idx", file_base);

        // read in the record offsets
        let file = File::open(&index_name).expect(&format!("Couldn't open data file {:?}", &index_name));
        let reader = BufReader::with_capacity(10 * 1_024 * 1_024, &file);
        let mut record_offsets: Vec<usize> = rmp_serde::decode::from_read(reader).expect(&format!("Error reading data from file {:?}", &index_name));

        // add additional offset, being the end of the file
        let metadata = std::fs::metadata(&data_name).expect(&format!("Couldn't get metadata from file {:?}", &data_name));
        record_offsets.push(metadata.len() as usize);

        // memory map the file
        let file = File::open(&data_name).expect(&format!("Couldn't open data file {:?}", &data_name));
        
        let options = match access_type {
            AccessType::Sequential => MmapOptions::new()
                .huge(Some(22))                 // 2^21 == 4MB pages
                .no_reserve_swap().clone(),
            AccessType::Random => MmapOptions::new()
                .no_reserve_swap().clone(),
        };

        let mapped_file = unsafe {
            // Mmap::map(&file).expect("failed to memory map file")
            options.map(&file).expect("failed mapping file")
        };

        match access_type {
            AccessType::Sequential => {
                mapped_file.advise(Advice::Sequential).expect("failed to advise mmap of sequential");
                mapped_file.advise(Advice::HugePage).expect("failed to advise mmap of huge pages");
            },
            AccessType::Random => {
                mapped_file.advise(Advice::Random).expect("failed to advise mmap of random");
                mapped_file.advise(Advice::NoHugePage).expect("failed to advise mmap of no huge pages");
            }
        }

        Self {
            record_offsets,
            mapped_file,
            _marker: PhantomData,
        }
    }
}

impl<T> DataReader<T> for MpkDataReader<T>
where T: DeserializeOwned
{
    fn read(&self, index: usize) -> Result<Vec<T>, Box<dyn Error>> {
        if index >= self.len() {
            return Err("Index out of bounds".into());
        }

        let start = self.record_offsets[index];
        let end = self.record_offsets[index + 1];

        // Slice the mmap (looks like reading from RAM)
        let bytes = &self.mapped_file[start..end];

        // unpack using message pack
        let data = rmp_serde::from_slice(bytes)?;

        Ok(data)
    }

    fn len(&self) -> usize {
        self.record_offsets.len() - 1
    }
}




